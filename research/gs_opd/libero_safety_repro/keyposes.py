# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Object- and region-frame Keyposes with strict SE(3) frame conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


KEYPOSE_STAGES = (
    "pre_grasp",
    "grasp",
    "lift",
    "pre_place",
    "place",
    "release",
    "retreat",
)
VARIANT_COUNT = 5
BASE_FRAME = "robot_base"
POSE_UNIT = "m"
QUATERNION_ORDER = "wxyz"


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}_must_be_a_finite_vector_of_length_{size}")
    return result


@dataclass(frozen=True)
class VariantBounds:
    """Per-axis maximum translations in meters and rotations in radians."""

    translation_half_range_m: tuple[float, float, float]
    rotation_half_range_rad: tuple[float, float, float]

    def __post_init__(self) -> None:
        translation = _finite_vector(self.translation_half_range_m, 3, "translation_half_range_m")
        rotation = _finite_vector(self.rotation_half_range_rad, 3, "rotation_half_range_rad")
        if np.any(translation < 0.0) or np.any(rotation < 0.0):
            raise ValueError("variant_bounds_must_be_nonnegative")
        object.__setattr__(self, "translation_half_range_m", tuple(float(v) for v in translation))
        object.__setattr__(self, "rotation_half_range_rad", tuple(float(v) for v in rotation))


@dataclass(frozen=True)
class Keypose:
    """One tool pose relative to a named canonical object or target-region frame."""

    stage: str
    owner_role: str
    owner_frame_id: str
    tool_frame_id: str
    transform_owner_tool: np.ndarray
    gripper_intent: str

    def __post_init__(self) -> None:
        if self.stage not in KEYPOSE_STAGES:
            raise ValueError(f"unsupported_keypose_stage:{self.stage}")
        if self.owner_role not in ("target_object", "goal_region"):
            raise ValueError(f"unsupported_keypose_owner_role:{self.owner_role}")
        for field_name, value in (
            ("owner_frame_id", self.owner_frame_id),
            ("tool_frame_id", self.tool_frame_id),
            ("gripper_intent", self.gripper_intent),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name}_must_be_a_nonempty_string")
        transform = validate_transform(self.transform_owner_tool, "transform_owner_tool")
        transform.setflags(write=False)
        object.__setattr__(self, "transform_owner_tool", transform)


@dataclass(frozen=True)
class KeyposeVariant:
    """A reproducibly perturbed Keypose and its zero-based candidate index."""

    stage: str
    variant_id: int
    keypose: Keypose
    translation_delta_owner_m: tuple[float, float, float]
    rotation_delta_owner_xyz_rad: tuple[float, float, float]


@dataclass(frozen=True)
class BaseToolTarget:
    """A planner-tool target in the robot base frame, with frame provenance."""

    transform_base_tool: np.ndarray
    base_frame_id: str
    owner_role: str
    owner_frame_id: str
    tool_frame_id: str
    composition: str


# These small, stage-specific bounds are GS-OPD engineering choices. The paper
# states N=5 spatial variants but publishes neither their values nor a sampler.
DEFAULT_VARIANT_BOUNDS: Mapping[str, VariantBounds] = {
    "pre_grasp": VariantBounds((0.010, 0.010, 0.004), (0.035, 0.035, 0.070)),
    "grasp": VariantBounds((0.002, 0.002, 0.001), (0.012, 0.012, 0.025)),
    "lift": VariantBounds((0.008, 0.008, 0.012), (0.025, 0.025, 0.050)),
    "pre_place": VariantBounds((0.010, 0.010, 0.004), (0.035, 0.035, 0.070)),
    "place": VariantBounds((0.002, 0.002, 0.001), (0.012, 0.012, 0.025)),
    "release": VariantBounds((0.002, 0.002, 0.001), (0.012, 0.012, 0.025)),
    "retreat": VariantBounds((0.008, 0.008, 0.010), (0.025, 0.025, 0.050)),
}


def transform_from_pose(position_m: Any, quaternion_wxyz: Any) -> np.ndarray:
    """Build an SE(3) transform from meters and a unit WXYZ quaternion."""
    position = _finite_vector(position_m, 3, "position_m")
    quaternion = _finite_vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"quaternion_wxyz_must_be_unit_norm:{norm:.9g}")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def validate_transform(value: Any, name: str = "transform") -> np.ndarray:
    """Return a copy after checking a finite, right-handed rigid transform."""
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name}_must_be_a_finite_4x4_matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise ValueError(f"{name}_has_invalid_homogeneous_row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name}_rotation_is_not_orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name}_rotation_determinant_is_not_positive_one")
    return transform.copy()


def inverse_transform(value: Any) -> np.ndarray:
    """Invert a rigid transform without applying a generic matrix inverse."""
    transform = validate_transform(value)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def transform_to_pose(transform: Any) -> dict[str, list[float]]:
    """Serialize an SE(3) transform with explicit metric and WXYZ fields."""
    validated = validate_transform(transform)
    quaternion_xyzw = Rotation.from_matrix(validated[:3, :3]).as_quat()
    quaternion_wxyz = np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]]
    return {
        "position_m": validated[:3, 3].tolist(),
        "quaternion_wxyz": quaternion_wxyz.tolist(),
        "matrix": validated.tolist(),
    }


def keypose_to_base_tool(
    keypose: Keypose,
    *,
    transform_world_base: Any,
    transforms_world_owner: Mapping[str, Any],
) -> BaseToolTarget:
    """Convert ``T_owner_tool`` to ``T_base_tool`` using world-frame inputs.

    The owner pose must be the canonical target-object pose for object-owned
    phases, or the goal-region pose for region-owned phases. The tool frame is
    preserved verbatim, allowing callers to require an explicit tool adapter.
    """
    world_base = validate_transform(transform_world_base, "transform_world_base")
    if keypose.owner_frame_id not in transforms_world_owner:
        raise ValueError(f"world_pose_missing_for_owner_frame:{keypose.owner_frame_id}")
    world_owner = validate_transform(
        transforms_world_owner[keypose.owner_frame_id],
        f"transform_world_owner:{keypose.owner_frame_id}",
    )
    base_tool = inverse_transform(world_base) @ world_owner @ keypose.transform_owner_tool
    base_tool = validate_transform(base_tool, "transform_base_tool")
    return BaseToolTarget(
        transform_base_tool=base_tool,
        base_frame_id=BASE_FRAME,
        owner_role=keypose.owner_role,
        owner_frame_id=keypose.owner_frame_id,
        tool_frame_id=keypose.tool_frame_id,
        composition=(
            f"inverse(T_world_{BASE_FRAME}) @ T_world_{keypose.owner_frame_id} "
            f"@ T_{keypose.owner_frame_id}_{keypose.tool_frame_id}"
        ),
    )


def generate_variants(
    keypose: Keypose,
    *,
    seed: int,
    bounds: VariantBounds | None = None,
) -> tuple[KeyposeVariant, ...]:
    """Generate exactly five seeded local-frame spatial variants.

    Translation offsets and XYZ Euler perturbations are sampled uniformly in
    the owner frame. A stage-specific SeedSequence keeps the result independent
    of call order across stages.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed_must_be_a_nonnegative_integer")
    if bounds is None:
        bounds = DEFAULT_VARIANT_BOUNDS[keypose.stage]
    if not isinstance(bounds, VariantBounds):
        raise ValueError("bounds_must_be_a_VariantBounds_instance")
    stage_index = KEYPOSE_STAGES.index(keypose.stage)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), stage_index]))
    translation_range = np.asarray(bounds.translation_half_range_m, dtype=np.float64)
    rotation_range = np.asarray(bounds.rotation_half_range_rad, dtype=np.float64)
    results = []
    for variant_id in range(VARIANT_COUNT):
        translation_delta = rng.uniform(-translation_range, translation_range)
        rotation_delta = rng.uniform(-rotation_range, rotation_range)
        delta_rotation = Rotation.from_euler("xyz", rotation_delta).as_matrix()
        transform = keypose.transform_owner_tool.copy()
        transform[:3, 3] += translation_delta
        transform[:3, :3] = delta_rotation @ transform[:3, :3]
        variant_pose = Keypose(
            stage=keypose.stage,
            owner_role=keypose.owner_role,
            owner_frame_id=keypose.owner_frame_id,
            tool_frame_id=keypose.tool_frame_id,
            transform_owner_tool=transform,
            gripper_intent=keypose.gripper_intent,
        )
        results.append(
            KeyposeVariant(
                stage=keypose.stage,
                variant_id=variant_id,
                keypose=variant_pose,
                translation_delta_owner_m=tuple(float(v) for v in translation_delta),
                rotation_delta_owner_xyz_rad=tuple(float(v) for v in rotation_delta),
            )
        )
    return tuple(results)


def sample_variant_bundle(
    variants_by_stage: Mapping[str, tuple[KeyposeVariant, ...]],
    *,
    seed: int,
) -> dict[str, KeyposeVariant]:
    """Choose one of five variants per required stage deterministically."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed_must_be_a_nonnegative_integer")
    if set(variants_by_stage) != set(KEYPOSE_STAGES):
        raise ValueError("variant_library_must_contain_all_seven_keypose_stages")
    selected = {}
    for stage_index, stage in enumerate(KEYPOSE_STAGES):
        variants = variants_by_stage[stage]
        if len(variants) != VARIANT_COUNT or any(item.stage != stage for item in variants):
            raise ValueError(f"stage_must_have_exactly_five_matching_variants:{stage}")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x4B50, stage_index]))
        selected[stage] = variants[int(rng.integers(0, VARIANT_COUNT))]
    return selected
