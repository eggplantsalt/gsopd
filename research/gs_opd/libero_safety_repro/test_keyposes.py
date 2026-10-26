# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Contract tests for object-frame Keyposes and deterministic variants."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from research.gs_opd.libero_safety_repro.keyposes import (
    KEYPOSE_STAGES,
    Keypose,
    VariantBounds,
    generate_variants,
    inverse_transform,
    keypose_to_base_tool,
    sample_variant_bundle,
    transform_from_pose,
    transform_to_pose,
    validate_transform,
)


def _transform(position, xyz_euler):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", xyz_euler).as_matrix()
    result[:3, 3] = position
    return result


def _keypose(stage="grasp", owner="bowl_frame", role="target_object"):
    return Keypose(
        stage=stage,
        owner_role=role,
        owner_frame_id=owner,
        tool_frame_id="panda_hand",
        transform_owner_tool=_transform([0.08, -0.03, 0.12], [0.2, -0.4, 0.7]),
        gripper_intent="close" if stage == "grasp" else "hold",
    )


def test_pose_conversion_uses_meters_and_wxyz_quaternion():
    rotation = Rotation.from_euler("xyz", [0.4, -0.1, 1.2])
    quaternion_xyzw = rotation.as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    transform = transform_from_pose([0.21, -0.14, 0.63], quaternion_wxyz)
    np.testing.assert_allclose(transform[:3, :3], rotation.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(transform[:3, 3], [0.21, -0.14, 0.63], atol=1e-12)
    record = transform_to_pose(transform)
    assert set(record) == {"position_m", "quaternion_wxyz", "matrix"}
    rebuilt = transform_from_pose(record["position_m"], record["quaternion_wxyz"])
    np.testing.assert_allclose(rebuilt, transform, atol=1e-12)


def test_inverse_is_a_rigid_inverse_and_preserves_orientation():
    transform = _transform([0.42, -0.17, 0.31], [0.63, -0.22, 1.04])
    inverse = inverse_transform(transform)
    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse[:3, :3], transform[:3, :3].T, atol=1e-12)


def test_object_or_region_to_base_tool_uses_full_noncommuting_se3_composition():
    world_base = _transform([0.3, -0.2, 0.4], [0.31, -0.27, 0.18])
    world_object = _transform([-0.1, 0.5, 0.2], [-0.41, 0.16, 0.83])
    object_tool = _transform([0.08, -0.03, 0.12], [0.2, -0.4, 0.7])
    keypose = Keypose(
        stage="grasp",
        owner_role="target_object",
        owner_frame_id="scene_object_7",
        tool_frame_id="panda_hand",
        transform_owner_tool=object_tool,
        gripper_intent="close",
    )
    result = keypose_to_base_tool(
        keypose,
        transform_world_base=world_base,
        transforms_world_owner={"scene_object_7": world_object},
    )
    expected = inverse_transform(world_base) @ world_object @ object_tool
    np.testing.assert_allclose(result.transform_base_tool, expected, atol=1e-12)
    np.testing.assert_allclose(
        world_base @ result.transform_base_tool,
        world_object @ keypose.transform_owner_tool,
        atol=1e-12,
    )
    assert result.base_frame_id == "robot_base"
    assert result.owner_role == "target_object"
    assert result.owner_frame_id == "scene_object_7"
    assert result.tool_frame_id == "panda_hand"
    assert result.composition == (
        "inverse(T_world_robot_base) @ T_world_scene_object_7 "
        "@ T_scene_object_7_panda_hand"
    )


def test_goal_region_frame_is_distinct_from_target_object_frame():
    region_pose = _transform([0.22, 0.09, 0.76], [0.0, 0.0, -0.5])
    keypose = _keypose("place", "plate_region", "goal_region")
    result = keypose_to_base_tool(
        keypose,
        transform_world_base=np.eye(4),
        transforms_world_owner={"plate_region": region_pose},
    )
    np.testing.assert_allclose(
        result.transform_base_tool,
        region_pose @ keypose.transform_owner_tool,
        atol=1e-12,
    )
    assert result.owner_role == "goal_region"
    assert result.owner_frame_id != "bowl_frame"


def test_variant_generation_is_fixed_seed_and_call_order_independent():
    poses = {stage: _keypose(stage) for stage in KEYPOSE_STAGES}
    forward = {stage: generate_variants(poses[stage], seed=431) for stage in KEYPOSE_STAGES}
    reverse = {stage: generate_variants(poses[stage], seed=431) for stage in reversed(KEYPOSE_STAGES)}
    for stage in KEYPOSE_STAGES:
        assert len(forward[stage]) == 5
        for first, second in zip(forward[stage], reverse[stage], strict=True):
            np.testing.assert_array_equal(first.keypose.transform_owner_tool, second.keypose.transform_owner_tool)
            assert first.variant_id == second.variant_id


def test_variant_set_is_reproducible_distinct_and_within_declared_bounds():
    base = _keypose()
    bounds = VariantBounds((0.01, 0.02, 0.03), (0.1, 0.2, 0.3))
    variants = generate_variants(base, seed=17, bounds=bounds)
    again = generate_variants(base, seed=17, bounds=bounds)
    assert len(variants) == 5
    assert len({tuple(np.round(item.keypose.transform_owner_tool[:3, 3], 12)) for item in variants}) == 5
    for first, second in zip(variants, again, strict=True):
        np.testing.assert_array_equal(first.keypose.transform_owner_tool, second.keypose.transform_owner_tool)
        assert np.all(np.abs(first.translation_delta_owner_m) <= np.asarray(bounds.translation_half_range_m))
        assert np.all(np.abs(first.rotation_delta_owner_xyz_rad) <= np.asarray(bounds.rotation_half_range_rad))
        validate_transform(first.keypose.transform_owner_tool)


def test_variant_sampling_selects_one_valid_candidate_per_stage_deterministically():
    library = {
        stage: generate_variants(_keypose(stage), seed=100)
        for stage in KEYPOSE_STAGES
    }
    first = sample_variant_bundle(library, seed=9001)
    second = sample_variant_bundle(library, seed=9001)
    assert tuple(first) == KEYPOSE_STAGES
    for stage in KEYPOSE_STAGES:
        assert first[stage].variant_id == second[stage].variant_id
        assert 0 <= first[stage].variant_id < 5


@pytest.mark.parametrize(
    "position, quaternion, message",
    [
        ([0.0, 0.0], [1.0, 0.0, 0.0, 0.0], "position_m_must_be_a_finite_vector_of_length_3"),
        ([0.0, 0.0, np.nan], [1.0, 0.0, 0.0, 0.0], "position_m_must_be_a_finite_vector_of_length_3"),
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], "quaternion_wxyz_must_be_unit_norm"),
        ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], "quaternion_wxyz_must_be_unit_norm"),
    ],
)
def test_pose_constructor_rejects_invalid_vectors(position, quaternion, message):
    with pytest.raises(ValueError, match=message):
        transform_from_pose(position, quaternion)


@pytest.mark.parametrize("transform, message", [
    (np.eye(3), "must_be_a_finite_4x4_matrix"),
    (np.full((4, 4), np.nan), "must_be_a_finite_4x4_matrix"),
    (np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]), "invalid_homogeneous_row"),
    (np.diag([2.0, 1.0, 1.0, 1.0]), "rotation_is_not_orthonormal"),
    (np.diag([-1.0, 1.0, 1.0, 1.0]), "rotation_determinant_is_not_positive_one"),
])
def test_transform_validation_rejects_non_se3_inputs(transform, message):
    with pytest.raises(ValueError, match=message):
        validate_transform(transform)


def test_keypose_rejects_unknown_stage_role_and_missing_owner_pose():
    with pytest.raises(ValueError, match="unsupported_keypose_stage"):
        Keypose("transport", "target_object", "obj", "panda_hand", np.eye(4), "hold")
    with pytest.raises(ValueError, match="unsupported_keypose_owner_role"):
        Keypose("grasp", "robot", "obj", "panda_hand", np.eye(4), "close")
    with pytest.raises(ValueError, match="world_pose_missing_for_owner_frame"):
        keypose_to_base_tool(
            _keypose(),
            transform_world_base=np.eye(4),
            transforms_world_owner={},
        )
