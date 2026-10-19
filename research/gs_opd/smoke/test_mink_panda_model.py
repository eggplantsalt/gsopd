# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np

from research.gs_opd.smoke.mink_panda_model import (
    derive_eef_target,
    model_hash_compatibility,
    quaternion_rotate_wxyz,
    select_archived_routes,
)


def _candidate(candidate_id, direction, accepted=True):
    candidate = {
        "candidate_id": candidate_id,
        "direction": direction,
        "budget_status": "accepted" if accepted else "rejected",
        "physical_tracking_performed": accepted,
        "reference_path": [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        "path_length_m": 0.1,
    }
    return candidate


class MinkPandaModelTests(unittest.TestCase):
    def test_world_frame_quaternion_transform_preserves_v5_rigid_offset(self):
        sqrt_half = np.sqrt(0.5)
        position, quaternion = derive_eef_target(
            [1.0, 2.0, 3.0],
            [sqrt_half, 0.0, 0.0, sqrt_half],
            [0.1, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(position, [1.0, 2.1, 3.0], atol=1e-12)
        np.testing.assert_allclose(
            quaternion, [sqrt_half, 0.0, 0.0, sqrt_half], atol=1e-12
        )

    def test_quaternion_rotation_uses_world_wxyz_convention(self):
        sqrt_half = np.sqrt(0.5)
        rotated = quaternion_rotate_wxyz(
            [sqrt_half, 0.0, 0.0, sqrt_half], [1.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)

    def test_selects_only_three_budget_accepted_archived_routes(self):
        routes = [
            _candidate("route-a", "left"),
            _candidate("route-b", "right"),
            _candidate("route-c", "above"),
            _candidate("route-d", "left", accepted=False),
        ]
        archive = {"decision_step": 55, "tracking": {"candidates": routes}}
        selected = select_archived_routes(archive)
        self.assertEqual(
            [route["candidate_id"] for route in selected],
            ["route-a", "route-b", "route-c"],
        )

    def test_route_selection_rejects_missing_or_duplicate_direction(self):
        archive = {
            "decision_step": 55,
            "tracking": {
                "candidates": [
                    _candidate("route-a", "left"),
                    _candidate("route-b", "left"),
                    _candidate("route-c", "above"),
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "directions_incomplete"):
            select_archived_routes(archive)

    def test_model_signature_mismatch_is_incompatible(self):
        compatible, reason = model_hash_compatibility("a" * 64, "b" * 64)
        self.assertFalse(compatible)
        self.assertEqual(reason, "compiled_model_signature_mismatch")


if __name__ == "__main__":
    unittest.main()
