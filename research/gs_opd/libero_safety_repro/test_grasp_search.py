# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Test contact accounting and the local grasp screening contract."""

import copy

import numpy as np
import pytest

from research.gs_opd.libero_safety_repro.contact_diagnosis import aggregate_pairs
from research.gs_opd.libero_safety_repro.execute_grasp import (
    actuator_pair,
    classify_contacts,
)
from research.gs_opd.libero_safety_repro.grasp_search import (
    first_bilateral_sample,
    local_candidates,
)


def source():
    return {
        "collision_scene_export": {
            "geometry_inventory": {
                "status": "complete",
                "geometries": [
                    {
                        "geom_id": 1,
                        "geom_name": "pad",
                        "body_id": 10,
                        "body_name": "tip",
                        "category": "robot",
                    },
                    {
                        "geom_id": 2,
                        "geom_name": "target",
                        "body_id": 20,
                        "body_name": "object",
                        "category": "scene",
                    },
                ],
            }
        },
        "target_context": {
            "fingertip_pad_geometries": {"left": [{"geom_id": 1}]},
            "target_collision_geometries": [{"geom_id": 2}],
        },
    }


def contact():
    return {
        "geom_ids": [1, 2],
        "geom_names": ["pad", "target"],
        "body_ids": [10, 20],
        "distance_m": -0.001,
        "allowed": True,
    }


def test_manifold_points_are_not_samples_or_pairs():
    row = {
        "contacts": [contact(), contact()],
        "gripper_joint_positions_rad": [0.01, -0.01],
    }
    (pair,) = aggregate_pairs([row, row], source())
    assert pair["contact_record_count"] == 4
    assert pair["sample_count"] == 2
    assert pair["deepest_penetration_m"] == 0.001
    assert not pair["robot_self_collision"]
    assert pair["identities"][0]["finger_pad_side"] == "left"


def test_identity_mismatch_rejected():
    c = contact()
    c["body_ids"][0] = 99
    with pytest.raises(ValueError, match="identity"):
        aggregate_pairs([{"contacts": [c]}], source())


def test_incomplete_inventory_rejected():
    s = source()
    s["collision_scene_export"]["geometry_inventory"]["status"] = "partial"
    with pytest.raises(ValueError, match="complete"):
        aggregate_pairs([], s)


def test_local_offset_uses_tool_axes():
    reference = np.array(
        [[0.0, -1, 0, 0.2], [1, 0, 0, 0.3], [0, 0, 1, 0.4], [0, 0, 0, 1]]
    )
    (candidate,) = local_candidates(reference, [[0.01, 0, 0]])
    np.testing.assert_allclose(candidate[:3, 3], [0.2, 0.31, 0.4])
    np.testing.assert_allclose(candidate[:3, :3], reference[:3, :3])
    np.testing.assert_allclose(reference[:3, 3], [0.2, 0.3, 0.4])


@pytest.mark.parametrize(
    "offsets", [[], [[0, 0, 0]] * 13, [[float("nan"), 0, 0]], [[0, 0]]]
)
def test_invalid_candidate_inputs(offsets):
    with pytest.raises(ValueError):
        local_candidates(np.eye(4), offsets)


def closure():
    return {
        "status": "complete",
        "physical_state_unchanged": True,
        "states": [
            {
                "contacts": [],
                "illegal_contact_count": 0,
                "gripper_joint_positions_rad": [0.01, -0.01],
            },
            {
                "contacts": [
                    {"geom_ids": [1, 3], "allowed": True},
                    {"geom_ids": [2, 3], "allowed": True},
                ],
                "illegal_contact_count": 0,
                "gripper_joint_positions_rad": [0.002, -0.002],
            },
            {
                "contacts": [],
                "illegal_contact_count": 1,
                "gripper_joint_positions_rad": [0, 0],
            },
        ],
    }


def test_bilateral_contact_precedes_forced_zero_overlap():
    result = first_bilateral_sample(closure(), {1: "left", 2: "right"})
    assert result["accepted"] and result["sample_index"] == 1
    assert result["reason"] == "static_bilateral_geometry_only"


def test_nonpad_contact_before_bilateral_is_rejected():
    audit = copy.deepcopy(closure())
    audit["states"][0]["illegal_contact_count"] = 1
    assert not first_bilateral_sample(audit, {1: "left", 2: "right"})["accepted"]


def test_one_pad_does_not_count_as_bilateral():
    audit = closure()
    audit["states"] = audit["states"][:2]
    assert not first_bilateral_sample(audit, {1: "left"})["accepted"]


def test_missing_monitoring_is_rejected():
    assert not first_bilateral_sample({"status": "partial"}, {})["accepted"]


def test_physical_pad_permission_does_not_allow_nonpad_or_self_contact():
    geoms = {
        1: {"geom_name": "pad", "category": "robot"},
        2: {"geom_name": "finger_mesh", "category": "robot"},
        3: {"geom_name": "target", "category": "scene"},
        4: {"geom_name": "obstacle", "category": "scene"},
    }

    def sample(a, b):
        return {
            "status": "contact",
            "contacts": [
                {
                    "geom1": {"id": a, "name": geoms[a]["geom_name"]},
                    "geom2": {"id": b, "name": geoms[b]["geom_name"]},
                }
            ],
        }

    assert not classify_contacts(sample(1, 3), geoms, {1}, {3}, True)["illegal"]
    assert classify_contacts(sample(2, 3), geoms, {1}, {3}, True)["illegal"]
    assert classify_contacts(sample(1, 2), geoms, {1}, {3}, True)["illegal"]
    assert classify_contacts(sample(3, 4), geoms, {1}, {3}, True, {4})["illegal"]
    assert not classify_contacts({"status": "unavailable"}, geoms, {1}, {3}, True)[
        "complete"
    ]


def test_initial_and_formatted_panda_actuator_state():
    np.testing.assert_array_equal(actuator_pair([0.0]), [0.0, 0.0])
    np.testing.assert_array_equal(actuator_pair([-0.5, 0.5]), [-0.5, 0.5])
    with pytest.raises(ValueError):
        actuator_pair([0.5])
