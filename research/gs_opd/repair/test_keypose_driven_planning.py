import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.gs_opd.repair.keypose_driven_planning import (
    extract_keypose_experience,
    pose_matrix,
    retarget_keypose_sequence,
    select_keypose_suffix,
)


class KeyposeDrivenPlanningTests(unittest.TestCase):
    def _fixture(self, root: Path, *, success=True):
        object_id = "moka_pot_1"
        phases = [
            "lift_to_clearance", "high_west_clearance", "high_south_corridor",
            "high_yaw_alignment", "southwest_target_overhead", "slow_descend",
            "open_release", "post_release_grasp_axis_withdrawal",
        ]
        records = []
        pre_robot = []
        post_robot = []
        pre_objects = []
        post_objects = []
        for index, phase in enumerate(phases):
            z = 0.5 + 0.02 * index
            robot = {
                "robot_index": 0,
                "eef_pos": [0.1 + 0.01 * index, 0.2, z + 0.1],
                "eef_quat": [1.0, 0.0, 0.0, 0.0],
                "joint_qpos": [0.0] * 7,
                "joint_qvel": [0.0] * 7,
                "gripper_qpos": [0.0, 0.0],
            }
            obj = {
                f"objects_dict:{object_id}": {
                    "root_pos": [0.1 + 0.01 * index, 0.2, z],
                    "root_quat": [1.0, 0.0, 0.0, 0.0],
                }
            }
            pre_robot.append(json.dumps(robot))
            post_robot.append(json.dumps(robot))
            pre_objects.append(json.dumps(obj))
            post_objects.append(json.dumps(obj))
            records.append({
                "episode_timestep_before": index,
                "episode_timestep_after": index + 1,
                "target_object": object_id,
                "phase": phase,
                "action_source": "expert_osc",
                "physical_state_ref": {"archive": str(root / "trace.npz"), "row": index},
            })
        trace_npz = root / "trace.npz"
        np.savez(
            trace_npz,
            pre_robot_state_json=np.asarray(pre_robot),
            post_robot_state_json=np.asarray(post_robot),
            pre_all_object_states_json=np.asarray(pre_objects),
            post_all_object_states_json=np.asarray(post_objects),
        )
        trace_jsonl = root / "trace.jsonl"
        trace_jsonl.write_text("".join(json.dumps(row) + "\n" for row in records))
        acceptance = {
            "schema_version": "gs_opd_l0t2_complete_task_success_report.v1",
            "task": "LIBERO-Safety obstacle_avoidance/L0T2/state0",
            "instruction": "put both moka pots on the stove",
            "task_success": success,
            "success_criteria": {"native_predicates_all_true": success},
            "episode_timestep": len(records),
            "trajectory": {"action_count": len(records)},
            "bridge_last_reward": 1.0 if success else 0.0,
            "bridge_last_done": success,
            "final_native_goal_values": [{
                "predicate": ["on", object_id, "flat_stove_1_cook_region"],
                "is_native_goal": True,
                "value": success,
            }],
            "safety_label": {"safety_success": False, "reason": "unsafe contact is not safety certification"},
        }
        acceptance_path = root / "acceptance.json"
        acceptance_path.write_text(json.dumps(acceptance))
        return trace_jsonl, acceptance_path

    def test_extracts_actual_physical_keyposes_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, acceptance = self._fixture(root)
            region = pose_matrix([0.0, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
            experience = extract_keypose_experience(
                trace,
                acceptance,
                object_id="moka_pot_1",
                predicate=["on", "moka_pot_1", "flat_stove_1_cook_region"],
                source_region_world=region,
            )
        self.assertEqual(len(experience["keypose_sequence"]), 9)
        self.assertEqual(experience["keypose_sequence"][0]["name"], "grasp")
        self.assertEqual(experience["keypose_sequence"][0]["frame"], "object")
        self.assertEqual(experience["keypose_sequence"][-1]["name"], "retreat")
        self.assertEqual(experience["keypose_sequence"][-1]["frame"], "region")
        self.assertTrue(experience["source"]["native_task_success"])
        self.assertFalse(experience["source"]["safety_success"])
        self.assertFalse(experience["source"]["source_actions_replayed"])

    def test_refuses_incomplete_or_unsuccessful_source_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, acceptance_path = self._fixture(root, success=False)
            with self.assertRaisesRegex(ValueError, "not_the_verified_GS_OPD_L0T2_native_success"):
                extract_keypose_experience(
                    trace,
                    acceptance_path,
                    object_id="moka_pot_1",
                    predicate=["on", "moka_pot_1", "flat_stove_1_cook_region"],
                    source_region_world=np.eye(4),
                )

    def test_retargets_object_and_region_frames_to_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace, acceptance = self._fixture(root)
            source_region = pose_matrix([0.0, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
            experience = extract_keypose_experience(
                trace,
                acceptance,
                object_id="moka_pot_1",
                predicate=["on", "moka_pot_1", "flat_stove_1_cook_region"],
                source_region_world=source_region,
            )
        held = pose_matrix([0.4, -0.2, 0.7], [1.0, 0.0, 0.0, 0.0])
        eef = pose_matrix([0.4, -0.2, 0.8], [1.0, 0.0, 0.0, 0.0])
        destination_region = pose_matrix([-0.3, 0.15, 0.35], [1.0, 0.0, 0.0, 0.0])
        destination_object = pose_matrix([-0.27, 0.14, 0.46], [1.0, 0.0, 0.0, 0.0])
        relocated = retarget_keypose_sequence(
            experience,
            held_object_world=held,
            current_eef_world=eef,
            destination_region_world=destination_region,
            destination_object_world=destination_object,
        )
        self.assertEqual([row["name"] for row in relocated], [row["name"] for row in experience["keypose_sequence"]])
        self.assertEqual(relocated[-2]["object_pose_world"]["position_m"], destination_object[:3, 3].tolist())
        self.assertFalse(any(row["source_actions_replayed"] for row in relocated))

    def test_suffix_resume_does_not_replay_completed_keyposes(self):
        sequence = [{"name": name} for name in ("lift", "high_west", "high_south", "preplace", "place")]
        resumed = select_keypose_suffix(sequence, "high_south")
        self.assertEqual([row["name"] for row in resumed], ["preplace", "place"])
        with self.assertRaisesRegex(ValueError, "not_in_relocated_sequence"):
            select_keypose_suffix(sequence, "unknown")


if __name__ == "__main__":
    unittest.main()
