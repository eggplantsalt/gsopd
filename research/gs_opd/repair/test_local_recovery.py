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
"""CPU contract tests with a deterministic Bridge process-edge fake, not MuJoCo."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.gs_opd.data.trajectory_writer import TrajectoryWriter
from research.gs_opd.repair import local_recovery as recovery
from research.gs_opd.repair.near_far import query_geom_gap
from research.gs_opd.smoke.test_early_repair_s70_smoke import (
    _complete_monitor,
    _controller_context,
)
from research.gs_opd.smoke.test_pi05_on_policy_safety_collection import (
    _bridge_preflight_inventories,
    _load_helpers,
    _semantic_binding,
    _teacher_v2_object_only,
)


def context():
    bindings = {
        "object_1": _semantic_binding("object_1", 21, "carried-body", 5),
        "object_2": _semantic_binding("object_2", 22, "obstacle-body", 7),
    }
    geometry, gripper = _bridge_preflight_inventories(bindings)
    return {
        "teacher": _teacher_v2_object_only(),
        "object_id": "object_1",
        "semantic_results": bindings,
        "semantic_result": bindings["object_1"],
        "geometry_inventory": geometry,
        "gripper_inventory": gripper,
        "repair_context": _controller_context(),
        "task_object_binding": {
            "object_id": "object_1",
            "body_id": 21,
            "status": "matched_by_iou",
            "role": "target_object",
        },
        "safety_target_binding": {"object_id": "object_1", "body_id": 21},
    }


class BridgeFake:
    """Test-only point dynamics and synthetic contacts behind the wire protocol."""

    def __init__(self, ctx, base_gap=0.25):
        self.ctx = ctx
        self.base_gap = base_gap
        self.point = np.asarray([0.0, 0.0, 0.5])
        self.time = 0
        self.saved = {}
        self.requests = []
        self.pending = None
        self.collide_recovery = False
        self.incomplete_recovery = False
        self.bad_restore = False
        self.distance_unavailable = False
        self.disturb_next = False
        self.object_drop_per_step = 0.0

    def send(self, payload):
        self.pending = payload
        self.requests.append(copy.deepcopy(payload))

    def diagnostics(self):
        return {
            "mujoco_integration_state": [float(self.time), *self.point.tolist()],
            "robot_states": [
                {"eef_pos": self.point.tolist(), "joint_qpos": [0.0, 0.0]}
            ],
            "object_states": {
                "carried": {
                    "body_id": 21,
                    "root_pos": (
                        self.point
                        + [0, 0, -0.1 - self.object_drop_per_step * self.time]
                    ).tolist(),
                }
            },
            "kinematics_source": "deterministic_test_fixture",
        }

    def pack(self):
        image = np.full((2, 2, 3), self.time % 255, dtype=np.uint8)
        return {
            "reward": 0.0,
            "done": False,
            "instruction": "carry the item",
            "state": {
                "robot0_eef_pos": self.point.tolist(),
                "robot0_eef_quat": [0, 0, 0, 1],
                "robot0_gripper_qpos": [0, 0],
            },
            "images": {
                key: {"shape": image.shape, "dtype": "uint8", "data": image.tobytes()}
                for key in ("agentview_image", "robot0_eye_in_hand_image")
            },
            "snapshot_state": self.diagnostics(),
        }

    def gap(self):
        return float(self.base_gap + self.point[0] + 0.25 * abs(self.point[1]))

    def step(self, action):
        self.point += np.asarray(action[:3], dtype=np.float32).astype(np.float64) * 0.05
        if self.disturb_next:
            self.point[0] += 0.001
            self.disturb_next = False
        self.time += 1
        return self.pack()

    def recv(self):
        p = self.pending
        op = p["op"]
        if op == "snapshot":
            sid = f"snapshot-{len(self.saved)}"
            self.saved[sid] = (self.point.copy(), self.time)
            return {"snapshot_id": sid, "snapshot_state": self.diagnostics()}
        if op == "restore":
            point, time = self.saved[p["snapshot_id"]]
            self.point, self.time = point.copy(), time
            diagnostic = self.diagnostics()
            if self.bad_restore:
                diagnostic["mujoco_integration_state"][1] += 0.001
            return {
                "restored_snapshot_id": p["snapshot_id"],
                "snapshot_state": diagnostic,
            }
        if op == "step":
            return self.step(p["action"])
        if op == "geom_set_min_distance":
            gap = self.gap()
            root = self.point + [0, 0, -0.1]
            return {
                "status": "complete",
                "query_complete": not self.distance_unavailable,
                "distance_reliable": not self.distance_unavailable,
                "minimum_distance_m": gap,
                "minimum_pair": {
                    "geom_a": {"id": 5, "name": "geom_5", "body_id": 21},
                    "geom_b": {"id": 7, "name": "geom_7", "body_id": 22},
                    "closest_segment_world": [
                        *root.tolist(),
                        *(root - [gap, 0, 0]).tolist(),
                    ],
                },
            }
        if op == "branch_rollout":
            caller = self.point.copy(), self.time
            results = []
            for spec in p["candidates"]:
                point, time = self.saved[p["snapshot_id"]]
                self.point, self.time = point.copy(), time
                item = {
                    "candidate_id": spec["candidate_id"],
                    "status": "ok",
                    "requested_steps": 5,
                    "executed_steps": 5,
                    "start_state": self.diagnostics(),
                    "steps": [],
                }
                contact_steps = []
                for index, action in enumerate(spec["actions"]):
                    data = self.step(action)
                    item["steps"].append({
                        "step_index": index,
                        "action": action,
                        "reward": data["reward"],
                        "done": data["done"],
                        "state": self.diagnostics(),
                    })
                    if self.gap() < 0 or (
                        self.collide_recovery
                        and spec["candidate_id"].startswith("recovery")
                    ):
                        contact_steps.append(index)
                _complete_monitor(item)
                monitor = item["contact_monitoring"]
                if contact_steps:
                    for index in contact_steps:
                        monitor["control_steps"][index]["contact_samples"] = {
                            "contact": 25,
                            "no_contact": 0,
                            "unavailable": 0,
                        }
                    count = 25 * len(contact_steps)
                    monitor.update({
                        "contact_substeps": count,
                        "no_contact_substeps": 125 - count,
                        "events": [
                            {
                                "geom_pair": [
                                    {"id": 5, "name": "geom_5"},
                                    {"id": 7, "name": "geom_7"},
                                ],
                                "body_pair": [
                                    {"id": 21, "name": "carried-body"},
                                    {"id": 22, "name": "obstacle-body"},
                                ],
                                "classification": "scene_scene",
                                "robot_involved": False,
                                "first_control_step": contact_steps[0],
                                "first_physics_substep": 0,
                                "occurrence_count": count,
                                "contact_point_count": count,
                                "force_sample_count": count,
                                "force_unavailable_count": 0,
                                "max_force_norm": 1.0,
                                "first_contact_position_world": self.point.tolist(),
                                "first_contact_normal_geom1_to_geom2_world": [
                                    -1,
                                    0,
                                    0,
                                ],
                            }
                        ],
                    })
                if self.incomplete_recovery and spec["candidate_id"].startswith(
                    "recovery"
                ):
                    monitor["status"] = "unavailable"
                results.append(item)
            self.point, self.time = caller
            return {"main_state_restored": True, "candidates": results}
        raise AssertionError(p)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_helpers()
        self.ctx = context()
        self.bridge = BridgeFake(self.ctx)
        self.actions = np.zeros((5, 7), dtype=np.float32)
        self.actions[:, 0] = -0.74
        self.actions[:, 6] = -0.5

    def plan(self, actions=None, step=50):
        return self.helpers["run_on_policy_safety_collection_decision"](
            self.bridge,
            self.actions if actions is None else actions,
            "carry the item",
            step // 5,
            step,
            self.ctx,
        )

    def execute(self, decision, count=5):
        selected = self.helpers["select_verified_execution_chunk"](
            decision, np.asarray(decision["original_action_chunk_A0"])
        )
        self.assertTrue(selected["valid"], selected)
        actual, post = [], []
        for action in selected["actions"][:count]:
            self.bridge.send({
                "op": "step",
                "action": action.tolist(),
                "include_snapshot_state": True,
            })
            post.append(self.bridge.recv())
            actual.append(action)
            if (
                recovery.validate_actual_recovery_step(
                    decision, len(actual) - 1, action, post[-1]
                )["status"]
                != "complete"
            ):
                break
        gap = query_geom_gap(self.bridge, subject_geom_ids=[5], obstacle_geom_ids=[7])
        committed = recovery.commit_recovery_execution(
            self.ctx["recovery_state"], decision, actual, post, self.ctx, gap
        )
        return actual, post, committed

    def test_strong_approach_generates_direct_commands_beyond_old_relative_budget(self):
        directions = [
            {"name": name, "direction_world": direction}
            for name, direction in (
                ("nearest_geom_away", [1, 0, 0]),
                ("nearest_geom_lateral_positive", [0, 1, 0]),
                ("nearest_geom_lateral_negative", [0, -1, 0]),
            )
        ]
        state = recovery.RecoveryState(last_actual_action=self.actions[0].tolist())
        specs, rejected = recovery.generate_recovery_candidates(
            self.actions, directions, self.ctx["repair_context"], state
        )
        self.assertLessEqual(len(specs) + len(rejected), 36)
        stop = next(
            item
            for item in specs
            if item["diagnostics"]["candidate_type"] == "stop_approach"
        )
        self.assertGreater(
            stop["diagnostics"]["relative_A0_maximum_correction_m"], 0.032
        )
        self.assertGreater(stop["diagnostics"]["relative_A0_total_correction_m"], 0.08)
        self.assertEqual(stop["command_constraints"]["status"], "pass")
        np.testing.assert_array_equal(
            np.asarray(stop["actions"])[:, 3:], self.actions[:, 3:]
        )
        self.assertTrue(
            any(
                item["diagnostics"]["validation_policy"] == "legacy_relative_A0"
                for item in specs
            )
        )

    def test_controller_overflow_is_rejected_without_clipping(self):
        bad = self.actions.copy()
        bad[:, 1] = 2.0
        result = recovery.command_constraints(bad, self.ctx["repair_context"])
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["input_clipping_applied"])
        self.assertIn("out_of_bounds", result["reason"])

    def test_actual_command_slew_workspace_and_reference_frame_are_checked(self):
        jerky = self.actions.copy()
        jerky[:, :3] = [[1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1], [1, 1, 1]]
        self.assertEqual(
            recovery.command_constraints(jerky, self.ctx["repair_context"])["reason"],
            "recovery_absolute_command_slew_exceeded",
        )
        ctx = copy.deepcopy(self.ctx["repair_context"])
        ctx["robots"][0]["fields"]["input_ref_frame"] = "eef"
        self.assertEqual(
            recovery.command_constraints(self.actions, ctx)["status"], "unavailable"
        )
        self.bridge.send({"op": "snapshot"})
        snapshot = self.bridge.recv()
        side = self.actions.copy()
        side[:, :3] = [0, 0.5, 0]
        self.bridge.send({
            "op": "branch_rollout",
            "snapshot_id": snapshot["snapshot_id"],
            "candidates": [{"candidate_id": "workspace", "actions": side.tolist()}],
        })
        branch = self.bridge.recv()["candidates"][0]
        ctx = copy.deepcopy(self.ctx["repair_context"])
        ctx["robots"][0]["fields"]["position_limits"][1][1] = 0.01
        controller, _ = recovery._controller_fields(ctx)
        self.assertEqual(
            recovery._workspace_constraints(side, branch, controller)["status"], "fail"
        )

    def test_missing_task_states_and_object_detachment_are_rejected(self):
        self.bridge.send({"op": "snapshot"})
        snapshot = self.bridge.recv()
        safe = self.actions.copy()
        safe[:, :3] = 0
        self.bridge.send({
            "op": "branch_rollout",
            "snapshot_id": snapshot["snapshot_id"],
            "candidates": [{"candidate_id": "task", "actions": safe.tolist()}],
        })
        branch = self.bridge.recv()["candidates"][0]
        detached = copy.deepcopy(branch)
        detached["steps"][-1]["state"]["object_states"]["carried"]["root_pos"][1] += 0.1
        result = recovery.task_continuity(
            detached, branch, self.ctx["task_object_binding"]
        )
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["physical_grasp_confirmed"])
        del detached["steps"][0]["state"]["object_states"]
        result = recovery.task_continuity(
            detached, branch, self.ctx["task_object_binding"]
        )
        self.assertFalse(result["evaluation_complete"])
        broken = copy.deepcopy(self.ctx)
        broken["task_object_binding"]["body_id"] = 999
        with self.assertRaisesRegex(ValueError, "bound task object"):
            recovery._resolve_pair(broken)

    def test_warning_selects_real_bypass_and_commits_only_after_main_execution(self):
        before = self.bridge.diagnostics()
        decision = self.plan()
        self.assertEqual(
            decision["status"], "local_recovery_selected", decision["failure_reasons"]
        )
        self.assertEqual(self.bridge.diagnostics(), before)
        self.assertEqual(self.ctx["recovery_state"].phase, "normal")
        self.assertEqual(
            decision["recovery_search"]["selected_candidate"]["selection_category"], 0
        )
        _, _, committed = self.execute(decision)
        self.assertEqual(committed["status"], "complete", committed)
        self.assertEqual(self.ctx["recovery_state"].phase, "escape")
        self.assertNotEqual(self.ctx["recovery_state"].side, 0)
        self.assertTrue(committed["verified_safe_recovery_chunk"])
        self.assertFalse(committed["task_success_observed"])
        self.assertFalse(committed["high_quality_demonstration"])

    def test_safe_unwarned_student_A0_executes_without_recovery_search(self):
        away = self.actions.copy()
        away[:, 0] = 0.02
        decision = self.plan(away, 15)
        self.assertEqual(decision["status"], "a0_safe", decision["failure_reasons"])
        self.assertEqual(decision["near_far_risk_verdict"], "no_warning")
        self.assertFalse(decision["repair_attempted"])
        actual, _, committed = self.execute(decision)
        np.testing.assert_array_equal(actual, away)
        self.assertFalse(committed["verified_safe_recovery_chunk"])
        self.assertEqual(self.ctx["recovery_state"].phase, "normal")

    def test_normal_student_release_does_not_use_recovery_holding_heuristic(self):
        self.bridge.object_drop_per_step = 0.012
        away = self.actions.copy()
        away[:, 0] = 0.02
        decision = self.plan(away)
        self.assertEqual(decision["status"], "a0_safe", decision["failure_reasons"])
        _, _, committed = self.execute(decision)
        self.assertEqual(committed["status"], "complete")
        self.assertGreater(committed["actual_maximum_relative_position_change_m"], 0.05)
        self.assertFalse(committed["task_success_observed"])
        self.assertFalse(committed["high_quality_demonstration"])

    def test_violation_A0_can_enter_recovery_without_safe_early_repair_precondition(
        self,
    ):
        self.bridge.base_gap = 0.05
        decision = self.plan()
        self.assertEqual(decision["a0_verdict"], "violation")
        self.assertEqual(
            decision["status"], "local_recovery_selected", decision["failure_reasons"]
        )
        selected = self.helpers["select_verified_execution_chunk"](
            decision, self.actions
        )
        self.assertTrue(selected["valid"], selected)
        self.assertFalse(np.array_equal(selected["actions"], self.actions))
        self.assertEqual(
            decision["recovery_search"]["selected_candidate"][
                "physical_safety_verdict"
            ],
            "safe",
        )

    def test_compiler_obstacle_target_does_not_replace_the_carried_task_binding(self):
        self.ctx["object_id"] = "object_2"
        self.ctx["semantic_result"] = self.ctx["semantic_results"]["object_2"]
        self.ctx["safety_target_binding"] = {"object_id": "object_2", "body_id": 22}
        self.bridge.base_gap = 0.05
        decision = self.plan()
        self.assertEqual(decision["a0_verdict"], "violation")
        self.assertEqual(
            decision["status"], "local_recovery_selected", decision["failure_reasons"]
        )
        self.assertEqual(decision["recovery"]["binding_pair"], [21, 22])

    def test_forbidden_candidate_contacts_cannot_be_selected(self):
        self.bridge.collide_recovery = True
        decision = self.plan()
        self.assertTrue(decision["must_abort_episode"])
        self.assertFalse(
            self.helpers["select_verified_execution_chunk"](decision, self.actions)[
                "valid"
            ]
        )
        self.assertTrue(
            any(
                "forbidden_contact_in_recovery_candidate" in item["rejection_reasons"]
                for item in decision["recovery_search"]["candidate_results"]
            )
        )
        self.assertEqual(self.bridge.time, 0)

    def test_unknown_monitoring_or_restore_failure_stays_fail_closed(self):
        for field in ("incomplete_recovery", "bad_restore", "distance_unavailable"):
            with self.subTest(field=field):
                self.bridge = BridgeFake(self.ctx)
                self.ctx.pop("recovery_state", None)
                setattr(self.bridge, field, True)
                decision = self.plan()
                self.assertTrue(decision["must_abort_episode"])
                self.assertFalse(
                    self.helpers["select_verified_execution_chunk"](
                        decision, self.actions
                    )["valid"]
                )
                self.assertEqual(self.bridge.time, 0)

    def test_same_side_is_kept_across_chunks_and_clearance_requires_actual_history(
        self,
    ):
        first = self.plan()
        self.execute(first)
        side = self.ctx["recovery_state"].side
        second = self.plan(step=55)
        self.execute(second)
        self.assertEqual(self.ctx["recovery_state"].side, side)
        away = self.actions.copy()
        away[:, 0] = 0.04
        for step in (60, 65):
            clear = self.plan(away, step)
            self.assertEqual(clear["recovery"]["transition"], "confirm_clear")
            self.execute(clear)
        final = self.plan(away, 70)
        self.assertEqual(final["recovery"]["transition"], "exit")
        self.execute(final)
        self.assertEqual(self.ctx["recovery_state"].phase, "normal")
        self.assertEqual(self.ctx["recovery_state"].side, 0)

    def test_bypass_is_preferred_over_large_gap_retreat_and_retreat_is_bounded(self):
        def item(name, lateral, away, gap, mode):
            return {
                "candidate_id": name,
                "accepted": True,
                "lateral_progress_signed_m": lateral,
                "vertical_progress_m": 0.0,
                "away_progress_m": away,
                "endpoint_gap_m": gap,
                "actual_world_command_length_m": 0.02,
                "diagnostics": {"candidate_type": mode},
            }

        lateral = item("bypass", 0.005, 0.0, 0.025, "lateral")
        retreat = item("retreat", 0.0, 0.02, 0.5, "retreat")
        state = recovery.RecoveryState()
        chosen, _ = recovery.select_recovery_candidate([retreat, lateral], state)
        self.assertEqual(chosen["candidate_id"], "bypass")
        state.retreat_chunks = 2
        chosen, reasons = recovery.select_recovery_candidate([retreat], state)
        self.assertIsNone(chosen)
        self.assertEqual(reasons[0]["reason"], "bounded_retreat_budget_exhausted")
        retreat["lateral_progress_signed_m"] = 0.005
        chosen, reasons = recovery.select_recovery_candidate([retreat], state)
        self.assertIsNone(chosen)  # residual side motion must not disguise a retreat
        self.assertEqual(reasons[0]["reason"], "bounded_retreat_budget_exhausted")
        state.side = -1
        chosen, reasons = recovery.select_recovery_candidate([lateral], state)
        self.assertIsNone(chosen)
        self.assertEqual(reasons[0]["reason"], "opposite_recovery_side")

    def test_bounded_no_progress_stops_instead_of_looping(self):
        state = recovery.RecoveryState(phase="escape", stagnant_chunks=3)
        self.ctx["recovery_state"] = state
        decision = self.plan()
        self.assertTrue(decision["must_abort_episode"])
        self.assertIn(
            "bounded_recovery_progress_budget_exhausted",
            str(decision["failure_reasons"]),
        )
        self.assertFalse(
            self.helpers["select_verified_execution_chunk"](decision, self.actions)[
                "valid"
            ]
        )

    def test_episode_entry_budget_cannot_be_reset_by_exit_and_reentry(self):
        self.ctx["recovery_state"] = recovery.RecoveryState(recovery_entries=4)
        decision = self.plan()
        self.assertTrue(decision["must_abort_episode"])
        self.assertIn(
            "episode_recovery_budget_exhausted", str(decision["failure_reasons"])
        )

    def test_actual_state_mismatch_stops_without_claiming_success(self):
        decision = self.plan()
        self.bridge.disturb_next = True
        actual, _, committed = self.execute(decision)
        self.assertEqual(
            len(actual), 1
        )  # no second real action after the first mismatch
        self.assertEqual(committed["status"], "failed")
        self.assertTrue(committed["must_stop"])
        self.assertEqual(self.ctx["recovery_state"].phase, "stopped")

    def test_actual_reward_or_missing_integration_state_blocks_next_action(self):
        decision = self.plan()
        selected = self.helpers["select_verified_execution_chunk"](
            decision, self.actions
        )
        self.bridge.send({"op": "step", "action": selected["actions"][0].tolist()})
        data = self.bridge.recv()
        missing = copy.deepcopy(data)
        del missing["snapshot_state"]["mujoco_integration_state"]
        self.assertEqual(
            recovery.validate_actual_recovery_step(
                decision, 0, selected["actions"][0], missing
            )["status"],
            "failed",
        )
        data["reward"] = float("nan")
        self.assertEqual(
            recovery.validate_actual_recovery_step(
                decision, 0, selected["actions"][0], data
            )["status"],
            "failed",
        )

    def test_candidate_termination_and_lost_positive_reward_are_not_task_success(self):
        decision = self.plan()
        selected = decision["recovery_search"]["selected_candidate"]
        candidate = {
            "status": "ok",
            "requested_steps": 5,
            "executed_steps": 5,
            "start_state": decision["recovery_execution"]["start_state"],
            "steps": [
                {"state": state, "reward": 0.0, "done": False}
                for state in selected["branch_states"]
            ],
        }
        stopped = copy.deepcopy(candidate)
        stopped["steps"][-1]["done"] = True
        result = recovery.task_continuity(
            stopped, candidate, self.ctx["task_object_binding"]
        )
        self.assertEqual(result["reason"], "recovery_terminated_without_success_reward")
        baseline = copy.deepcopy(candidate)
        baseline["steps"][-1]["reward"] = 1.0
        result = recovery.task_continuity(
            candidate, baseline, self.ctx["task_object_binding"]
        )
        self.assertEqual(
            result["reason"], "recovery_lost_existing_positive_task_reward"
        )
        self.assertFalse(result["task_success_confirmed"])

    def test_labels_and_saved_actual_actions_match_and_partial_suffix_is_not_labeled(
        self,
    ):
        for count in (2, 5):
            self.ctx = context()
            self.bridge = BridgeFake(self.ctx)
            initial = self.bridge.pack()
            decision = self.plan()
            selection = self.helpers["select_verified_execution_chunk"](
                decision, self.actions
            )
            decision.update({
                "execution_selection_valid": True,
                "selected_action_chunk": selection["actions"].tolist(),
                "selected_action_source": selection["source"],
            })
            actual, post, result = self.execute(decision, count=count)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["verified_safe_recovery_chunk"], count == 5)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "actual.npz"
                writer = TrajectoryWriter(path, {})
                writer.add_decision(0, 0, initial, self.actions, decision)
                for index, (action, data) in enumerate(zip(actual, post)):
                    writer.add_step(index, 0, index, action, data)
                record = writer.finalize_decision_execution(0, decision)
                self.assertEqual(record["training_label_valid"], count == 5)
                self.assertEqual(
                    record["actual_execution"]["executed_action_steps"], count
                )
                writer.save(steps=count, chunks=1)
                with np.load(path, allow_pickle=False) as archive:
                    np.testing.assert_array_equal(
                        archive["predicted_actions"], self.actions
                    )
                    np.testing.assert_array_equal(archive["executed_actions"], actual)
                    self.assertFalse(
                        np.array_equal(archive["predicted_actions"][:count], actual)
                    )
                    saved = json.loads(str(archive["decision_safety_json"][0]))
                    self.assertFalse(saved["high_quality_task_demonstration"])
                    self.assertEqual(
                        int(archive["decision_agentview_image"][0, 0, 0, 0]), 0
                    )
                    self.assertEqual(
                        int(archive["post_agentview_image"][0, 0, 0, 0]), 1
                    )

    def test_writer_retains_true_mismatched_action_and_invalidates_its_label(self):
        decision = self.plan()
        selection = self.helpers["select_verified_execution_chunk"](
            decision, self.actions
        )
        decision.update({
            "execution_selection_valid": True,
            "selected_action_chunk": selection["actions"].tolist(),
        })
        with tempfile.TemporaryDirectory() as directory:
            writer = TrajectoryWriter(Path(directory) / "bad.npz", {})
            writer.add_decision(0, 0, self.bridge.pack(), self.actions, decision)
            changed = selection["actions"][0].copy()
            changed[1] += 0.01
            with self.assertRaisesRegex(ValueError, "actual transition retained"):
                writer.add_step(0, 0, 0, changed, self.bridge.pack())
            self.assertEqual(len(writer.steps), 1)
            actual = writer.finalize_decision_execution(0, decision)
            self.assertFalse(actual["training_label_valid"])
            self.assertFalse(
                actual["actual_execution"]["executed_actions_match_selected_prefix"]
            )

    def test_offline_archive_uses_saved_control_diagnostics_without_physics(self):
        directions, _ = recovery.probe.derive_clearance_directions(
            {
                "query_complete": True,
                "distance_reliable": True,
                "minimum_pair": {
                    "geom_a": {"id": 5, "body_id": 21},
                    "geom_b": {"id": 7, "body_id": 22},
                    "closest_segment_world": [0.1, 0, 0, 0, 0, 0],
                },
            },
            21,
            22,
            [0, 0, 1],
        )
        specs, rejected = recovery.probe.generate_clearance_candidates(
            self.actions,
            directions,
            self.ctx["repair_context"],
            include_brake_and_steer=True,
        )
        entries = [
            {key: value for key, value in spec.items() if key != "actions"}
            for spec in specs
        ]
        self.bridge.send({"op": "geom_set_min_distance"})
        pair = self.bridge.recv()
        record = {
            "original_action_chunk_A0": self.actions.tolist(),
            "early_repair_search": {
                "candidate_results": entries,
                "candidate_generation_rejections": rejected,
            },
            "near_far_result": {
                "distance_samples": [{**pair, "distance_available": True}]
            },
            "near_far_geom_bindings": {
                "subject_object_id": "object_1",
                "obstacle_object_id": "object_2",
            },
            "semantic_binding_validation": {
                "objects": {"object_1": {"body_id": 21}, "object_2": {"body_id": 22}}
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.npz"
            np.savez_compressed(
                path,
                decision_step_indices=[5],
                predicted_action_offsets=[0, 5],
                predicted_actions=self.actions,
                decision_safety_json=[json.dumps(record)],
                step_indices=[4],
                executed_actions=[self.actions[0]],
            )
            output = recovery.diagnose_archive_generation(path, [5])[0]
            self.assertEqual(output["status"], "generated_offline_only", output)
            self.assertFalse(output["physical_safety_verified"])
            self.assertLessEqual(
                output["generated_count"] + output["constraint_rejected_count"], 36
            )
            self.assertGreater(output["exceeds_old_relative_budget_count"], 0)

    def test_changed_selected_or_labeled_actions_are_rejected(self):
        decision = self.plan()
        changed = copy.deepcopy(decision)
        changed["recovery_search"]["selected_candidate"]["actions"][0][3] += 0.01
        result = self.helpers["select_verified_execution_chunk"](changed, self.actions)
        self.assertFalse(result["valid"])
        self.assertIn("rotation or gripper", result["reason"])
        changed = copy.deepcopy(decision)
        changed["training_action_chunk"][0][0] += 0.01
        self.assertFalse(
            self.helpers["select_verified_execution_chunk"](changed, self.actions)[
                "valid"
            ]
        )
        changed = copy.deepcopy(decision)
        changed["recovery_search"]["selected_candidate"]["physical_safety_verdict"] = (
            "unknown"
        )
        self.assertFalse(
            self.helpers["select_verified_execution_chunk"](changed, self.actions)[
                "valid"
            ]
        )


if __name__ == "__main__":
    unittest.main()
