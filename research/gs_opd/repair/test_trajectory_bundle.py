# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic geometry tests; no model, simulator, Bridge, or real scene claims."""

import copy
import io
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.gs_opd.repair import trajectory_bundle as planner


def sphere(object_id, center, radius):
    return {
        "object_id": object_id,
        "center_world_m": center,
        "radius_m": radius,
        "binding_verified": True,
        "enclosure_verified": True,
        "binding_source": "synthetic test identity, not a real scene binding",
        "enclosure_source": "synthetic full sphere, not an inferred mesh extent",
        "state_id": "synthetic-state",
    }


def scene():
    return {
        "frame": "world",
        "units": "m",
        "state_id": "synthetic-state",
        "world_up": [0, 0, 1],
        "carried": sphere("arbitrary-payload", [-0.4, 0, 0.5], 0.04),
        "obstacles": [sphere("arbitrary-obstacle", [0, 0, 0.5], 0.06)],
        "goal": {
            "position_world_m": [0.4, 0, 0.5],
            "position_semantics": "carried_sphere_center",
            "verified": True,
            "source": "synthetic desired payload center",
            "state_id": "synthetic-state",
        },
    }


class TrajectoryBundleTests(unittest.TestCase):
    def test_four_control_points_match_analytic_cubic_bezier(self):
        controls = np.asarray([[0, 0, 0], [1, 2, 0], [2, 1, 1], [3, 0, 2]], dtype=float)
        actual = planner.discretize_bspline(controls, 31)
        t = np.linspace(0, 1, 31)[:, None]
        expected = (
            (1 - t) ** 3 * controls[0]
            + 3 * (1 - t) ** 2 * t * controls[1]
            + 3 * (1 - t) * t**2 * controls[2]
            + t**3 * controls[3]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(actual[[0, -1]], controls[[0, -1]])

    def test_default_bundle_has_three_families_and_exact_current_to_goal_paths(self):
        input_scene = scene()
        original = copy.deepcopy(input_scene)
        result = planner.generate_trajectory_bundle(input_scene)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["candidate_count"], 18)
        self.assertEqual(result["screen_pass_count"], 18)
        self.assertEqual(input_scene, original)
        self.assertEqual(result, planner.generate_trajectory_bundle(input_scene))
        for direction in ("left", "right", "above"):
            self.assertEqual(
                sum(item["direction"] == direction for item in result["candidates"]), 6
            )
        for candidate in result["candidates"]:
            points = np.asarray(candidate["points_world_m"])
            np.testing.assert_array_equal(
                points[0], input_scene["carried"]["center_world_m"]
            )
            np.testing.assert_array_equal(
                points[-1], input_scene["goal"]["position_world_m"]
            )
            self.assertGreater(candidate["goal_side_projection_m"], 0)
            self.assertGreater(candidate["length_m"], 0.8)
            self.assertGreaterEqual(
                candidate["screening"]["continuous_gap_lower_bound_m"], 0.01
            )
            self.assertFalse(candidate["physical_safety_verified"])
            self.assertFalse(candidate["executable"])
        self.assertLess(result["straight_line_minimum_gap_m"], 0)

    def test_all_allowed_bundle_sizes_stay_within_budget(self):
        for count in range(12, 25):
            with self.subTest(count=count):
                result = planner.generate_trajectory_bundle(
                    scene(), planner.BundleConfig(candidate_count=count)
                )
                self.assertEqual(len(result["candidates"]), count)
                self.assertEqual(
                    len({c["candidate_id"] for c in result["candidates"]}), count
                )

    def test_direction_frame_follows_task_and_world_up(self):
        input_scene = scene()
        input_scene["goal"]["position_world_m"] = [-0.4, 0.8, 0.5]
        input_scene["obstacles"][0]["center_world_m"] = [-0.4, 0.4, 0.5]
        result = planner.generate_trajectory_bundle(input_scene)
        np.testing.assert_allclose(result["frame_axes"]["left"], [-1, 0, 0], atol=1e-12)
        for candidate in result["candidates"]:
            direction = np.asarray(candidate["direction_world"])
            if candidate["direction"] == "above":
                self.assertGreater(float(np.dot(direction, input_scene["world_up"])), 0)

    def test_aabb_sphere_contains_every_box_corner(self):
        lower, upper = np.asarray([-0.1, -0.2, -0.3]), np.asarray([0.1, 0.2, 0.3])
        bound = planner.sphere_from_aabb(lower, upper)
        corners = np.asarray([
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ])
        self.assertTrue(
            np.all(
                np.linalg.norm(corners - bound["center_world_m"], axis=1)
                <= bound["radius_m"]
            )
        )
        # A tiny box at large coordinates can round its midpoint to one corner.
        distant_lower = np.full(3, 1e12)
        distant_upper = np.nextafter(distant_lower, np.inf)
        distant = planner.sphere_from_aabb(distant_lower, distant_upper)
        self.assertLessEqual(
            float(np.linalg.norm(distant_upper - distant["center_world_m"])),
            distant["radius_m"],
        )
        with self.assertRaises(ValueError):
            planner.sphere_from_aabb(upper, lower)
        with self.assertRaises(ValueError):
            planner.sphere_from_aabb([0, 0, 0], [0, 0, 0])

    def test_carried_radius_and_clearance_are_included_in_every_sphere_check(self):
        result = planner.generate_trajectory_bundle(scene())
        for candidate in result["candidates"]:
            check = candidate["screening"]["checks"][0]
            self.assertAlmostEqual(check["combined_sphere_radius_m"], 0.10)
            points = np.asarray(candidate["points_world_m"])
            expected = np.linalg.norm(points - [0, 0, 0.5], axis=1).min() - 0.10
            self.assertAlmostEqual(check["minimum_sampled_gap_m"], expected)

    def test_between_sample_curve_collision_cannot_pass_point_only_check(self):
        controls = [[-0.1, 0, 0], [-0.1 / 3, 0, 0], [0.1 / 3, 0, 0], [0.1, 0, 0]]
        obstacles = [sphere("thin-obstacle", [0, 0, 0], 0.001)]
        config = planner.BundleConfig(
            clearance_m=0, minimum_curve_samples=4, maximum_curve_samples=4
        )
        _, screen, reasons = planner._screen_curve(controls, 0.001, obstacles, config)
        self.assertGreater(screen["minimum_sampled_gap_m"], 0)
        self.assertLess(screen["continuous_gap_lower_bound_m"], 0)
        self.assertFalse(screen["checks"][0]["passes_sphere_screen"])
        self.assertIn("curve_clearance_bound_inconclusive:thin-obstacle", reasons)
        self.assertFalse(screen["resolution_target_met"])

    def test_clearance_bound_is_below_dense_curve_minimum(self):
        result = planner.generate_trajectory_bundle(
            scene(), planner.BundleConfig(candidate_count=12)
        )
        for candidate in result["candidates"]:
            dense = planner.discretize_bspline(
                candidate["control_points_world_m"], 10001
            )
            true_sample_minimum = (
                float(np.linalg.norm(dense - [0, 0, 0.5], axis=1).min()) - 0.10
            )
            self.assertLessEqual(
                candidate["screening"]["continuous_gap_lower_bound_m"],
                true_sample_minimum,
            )

    def test_other_obstacles_can_reject_left_paths_without_choosing_a_winner(self):
        input_scene = scene()
        input_scene["obstacles"].append(sphere("second-obstacle", [0, 0.18, 0.5], 0.08))
        result = planner.generate_trajectory_bundle(input_scene)
        self.assertGreater(result["screen_rejected_count"], 0)
        self.assertGreater(result["screen_pass_count"], 0)
        self.assertNotIn("selected_candidate_id", result)
        rejected = [
            item for item in result["candidates"] if not item["passes_geometric_screen"]
        ]
        self.assertTrue(
            any(
                "second-obstacle" in reason
                for item in rejected
                for reason in item["rejection_reasons"]
            )
        )

    def test_invalid_start_or_goal_keeps_all_rejected_candidates(self):
        for field in ("start", "goal"):
            with self.subTest(field=field):
                input_scene = scene()
                obstacle = input_scene["obstacles"][0]
                if field == "start":
                    input_scene["carried"]["center_world_m"] = [0, 0, 0.5]
                else:
                    input_scene["goal"]["position_world_m"] = [0, 0, 0.5]
                result = planner.generate_trajectory_bundle(input_scene)
                self.assertEqual(result["screen_pass_count"], 0)
                self.assertEqual(len(result["candidates"]), 18)
                for item in result["candidates"]:
                    self.assertIn(
                        f"{field}_sphere_clearance_insufficient:{obstacle['object_id']}",
                        item["rejection_reasons"],
                    )

    def test_invalid_geometry_provenance_frames_and_goal_semantics_are_refused(self):
        mutations = [
            lambda s: s.update(frame="camera"),
            lambda s: s.update(units="mm"),
            lambda s: s.update(world_up=[0, 0, 0]),
            lambda s: s["carried"].update(binding_verified=False),
            lambda s: s["carried"].update(enclosure_verified=False),
            lambda s: s["carried"].update(enclosure_source=""),
            lambda s: s["obstacles"][0].update(state_id="stale-state"),
            lambda s: s["obstacles"][0].update(object_id=s["carried"]["object_id"]),
            lambda s: s["obstacles"][0].update(radius_m=float("nan")),
            lambda s: s["obstacles"][0].update(radius_m=-0.01),
            lambda s: s["carried"].update(center_world_m=[True, 0, 0]),
            lambda s: s["goal"].update(verified=False),
            lambda s: s["goal"].update(state_id="stale-state"),
            lambda s: s["goal"].update(position_semantics="body_origin"),
            lambda s: s["goal"].update(position_world_m=[0, float("inf"), 0]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                input_scene = scene()
                mutate(input_scene)
                with self.assertRaises(ValueError):
                    planner.generate_trajectory_bundle(input_scene)

    def test_degenerate_route_and_configuration_are_refused(self):
        for goal in ([-0.4, 0, 0.5], [-0.4, 0, 1.0]):
            input_scene = scene()
            input_scene["goal"]["position_world_m"] = goal
            with self.assertRaises(ValueError):
                planner.generate_trajectory_bundle(input_scene)
        for config in (
            planner.BundleConfig(candidate_count=11),
            planner.BundleConfig(candidate_count=25),
            planner.BundleConfig(candidate_count=True),
            planner.BundleConfig(clearance_m=float("nan")),
            planner.BundleConfig(clearance_error_m=0),
            planner.BundleConfig(minimum_curve_samples=3),
            planner.BundleConfig(minimum_curve_samples=100, maximum_curve_samples=10),
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    planner.generate_trajectory_bundle(scene(), config)

    def test_invalid_bspline_inputs_are_refused(self):
        for controls, count in (
            ([[0, 0, 0]] * 3, 10),
            ([[0, 0, float("nan")]] * 4, 10),
            ([[0, 0, 0]] * 4, True),
        ):
            with self.assertRaises(ValueError):
                planner.discretize_bspline(controls, count)

    def test_ids_are_metadata_and_not_geometry_selection_rules(self):
        input_scene = scene()
        input_scene["obstacles"][0]["center_world_m"] = [0, 0.04, 0.5]
        input_scene["obstacles"].append(sphere("z-second", [0, -0.04, 0.5], 0.06))
        original = planner.generate_trajectory_bundle(input_scene)
        renamed = copy.deepcopy(input_scene)
        renamed["carried"]["object_id"] = "payload-any-ID-98"
        renamed["obstacles"][0]["object_id"] = "z-first-renamed"
        renamed["obstacles"][1]["object_id"] = "a-second-renamed"
        result = planner.generate_trajectory_bundle(renamed)
        for left, right in zip(original["candidates"], result["candidates"]):
            self.assertEqual(left["points_world_m"], right["points_world_m"])
            self.assertEqual(
                left["passes_geometric_screen"], right["passes_geometric_screen"]
            )

    def test_json_and_svg_include_all_candidates_and_three_world_views(self):
        input_scene = scene()
        input_scene["obstacles"][0]["object_id"] = "obstacle<&>"
        result = planner.generate_trajectory_bundle(input_scene)
        with tempfile.TemporaryDirectory() as directory:
            paths = planner.save_trajectory_bundle(result, directory)
            saved = json.loads(Path(paths["json"]).read_text())
            self.assertEqual(saved, result)
            root = ET.fromstring(Path(paths["svg"]).read_text())
            namespace = {"s": "http://www.w3.org/2000/svg"}
            self.assertEqual(
                {node.attrib["data-view"] for node in root.findall("s:g", namespace)},
                {"XY", "XZ", "YZ"},
            )
            self.assertEqual(len(root.findall(".//s:polyline", namespace)), 18 * 3)
            self.assertIn("not physical safety", "".join(root.itertext()))
            bad = copy.deepcopy(result)
            bad["candidates"][0]["length_m"] = float("nan")
            with self.assertRaises(ValueError):
                planner.save_trajectory_bundle(bad, Path(directory) / "bad")
            self.assertFalse((Path(directory) / "bad").exists())

    def test_offline_cli_saves_outputs_and_refuses_unverified_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, out = Path(directory) / "input.json", Path(directory) / "out"
            source.write_text(json.dumps(scene()))
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                status = planner.main([
                    "--scene-json",
                    str(source),
                    "--out-dir",
                    str(out),
                    "--candidate-count",
                    "12",
                ])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue())["candidate_count"], 12)
            self.assertTrue((out / "trajectory_bundle.svg").is_file())
            invalid = scene()
            invalid["carried"]["enclosure_verified"] = False
            source.write_text(json.dumps(invalid))
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                status = planner.main([
                    "--scene-json",
                    str(source),
                    "--out-dir",
                    str(Path(directory) / "bad"),
                ])
            self.assertEqual(status, 2)
            self.assertIn("unavailable", stderr.getvalue())
            source.write_text('{"units":"mm","units":"m"}')
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                status = planner.main([
                    "--scene-json",
                    str(source),
                    "--out-dir",
                    str(Path(directory) / "bad"),
                ])
            self.assertEqual(status, 2)
            self.assertIn("duplicate JSON key", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
