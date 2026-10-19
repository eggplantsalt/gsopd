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

import copy
import json
from pathlib import Path

import numpy as np

_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def _decode_image(bridge_data: dict, key: str) -> np.ndarray:
    image = bridge_data["images"][key]
    return (
        np.frombuffer(image["data"], dtype=np.dtype(image["dtype"]))
        .reshape(image["shape"])
        .copy()
    )


def _copy_state(bridge_data: dict, key: str) -> np.ndarray:
    return np.asarray(bridge_data["state"][key]).copy()


def _stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float32)
    return np.stack(values)


class TrajectoryWriter:
    """Collect one Bridge rollout and save it as a compressed NPZ file."""

    def __init__(self, output_path: Path, metadata: dict):
        self.output_path = Path(output_path)
        self.metadata = dict(metadata)
        self.decisions = []
        self.steps = []

    def add_decision(
        self,
        chunk_index: int,
        step_index: int,
        bridge_data: dict,
        predicted_actions: np.ndarray,
        safety_record: dict | None = None,
    ) -> None:
        """Record the raw observation and complete predicted action chunk."""
        decision = {
            "chunk_index": int(chunk_index),
            "step_index": int(step_index),
            "instruction": str(bridge_data["instruction"]),
            "predicted_actions": np.asarray(predicted_actions, dtype=np.float32).copy(),
        }
        for key in _CAMERA_KEYS:
            decision[key] = _decode_image(bridge_data, key)
        for key in _STATE_KEYS:
            decision[key] = _copy_state(bridge_data, key)
        if safety_record is not None:
            decision["safety_record"] = copy.deepcopy(safety_record)
        self.decisions.append(decision)

    def add_step(
        self,
        step_index: int,
        chunk_index: int,
        action_index: int,
        action: np.ndarray,
        bridge_data: dict,
    ) -> None:
        """Record one executed action and the Bridge observation it produced."""
        decision = next(
            (
                item
                for item in reversed(self.decisions)
                if item["chunk_index"] == chunk_index
            ),
            None,
        )
        record = decision.get("safety_record") if decision is not None else None
        mismatch = False
        if isinstance(record, dict) and record.get("execution_selection_valid") is True:
            selected = np.asarray(record.get("selected_action_chunk"), dtype=np.float32)
            actual = np.asarray(action, dtype=np.float32)
            if (
                not 0 <= action_index < len(selected)
                or actual.shape != (7,)
                or not np.array_equal(actual, selected[action_index])
            ):
                mismatch = True
        step = {
            "step_index": int(step_index),
            "chunk_index": int(chunk_index),
            "action_index": int(action_index),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(bridge_data["reward"]),
            "done": bool(bridge_data["done"]),
        }
        if "snapshot_state" in bridge_data:
            step["physical_state_json"] = json.dumps(
                bridge_data["snapshot_state"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        if "contacts" in bridge_data:
            step["contacts_json"] = json.dumps(
                bridge_data["contacts"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        for key in _CAMERA_KEYS:
            step[key] = _decode_image(bridge_data, key)
        for key in _STATE_KEYS:
            step[key] = _copy_state(bridge_data, key)
        self.steps.append(step)
        if mismatch:
            record["training_label_valid"] = False
            record["actual_action_mismatch"] = True
            raise ValueError(
                "Executed action does not match the selected verified action; actual transition retained"
            )

    def finalize_decision_execution(
        self, chunk_index: int, safety_record: dict
    ) -> dict:
        """Attach actual coverage; never label unexecuted suffixes as demonstrations."""
        decision = next(
            item
            for item in reversed(self.decisions)
            if item["chunk_index"] == chunk_index
        )
        record = copy.deepcopy(safety_record)
        transitions = [
            item for item in self.steps if item["chunk_index"] == chunk_index
        ]
        selected = np.asarray(record.get("selected_action_chunk"), dtype=np.float32)
        count = len(transitions)
        if selected.ndim != 2 or selected.shape[1:] != (7,):
            raise ValueError("Finalized decision has no valid selected action chunk")
        matches = count <= len(selected) and not any(
            step["action_index"] != index
            or not np.array_equal(step["action"], selected[index])
            for index, step in enumerate(transitions)
        )
        full = matches and count == len(selected)
        record["actual_execution"] = {
            "source": "main_environment_step_responses",
            "executed_action_steps": count,
            "requested_action_steps": len(selected),
            "full_selected_chunk_executed": full,
            "executed_actions_match_selected_prefix": matches,
            "unexecuted_suffix_is_training_label": False,
            "actual_rewards": [item["reward"] for item in transitions],
            "actual_dones": [item["done"] for item in transitions],
        }
        record["training_label_valid"] = (
            record.get("training_label_valid") is True and full
        )
        if not matches:
            record["actual_action_mismatch"] = True
        record["supervision_scope"] = (
            "verified_executed_action_chunk"
            if record["training_label_valid"]
            else "no_complete_training_label"
        )
        record["valid_supervision_action_steps"] = (
            count if record["training_label_valid"] else 0
        )
        record["task_success_observed_in_main_environment"] = any(
            item["reward"] > 0 for item in transitions
        )
        record["high_quality_task_demonstration"] = False
        decision["safety_record"] = record
        return record

    def save(self, *, steps: int, chunks: int) -> None:
        """Write decisions, executed transitions, index mappings, and metadata."""
        decision_count = len(self.decisions)
        step_count = len(self.steps)
        action_offsets = np.zeros(decision_count + 1, dtype=np.int64)
        for index, decision in enumerate(self.decisions):
            action_offsets[index + 1] = action_offsets[index] + len(
                decision["predicted_actions"]
            )
        if self.decisions:
            predicted_actions = np.concatenate(
                [item["predicted_actions"] for item in self.decisions], axis=0
            )
        else:
            predicted_actions = np.empty((0, 7), dtype=np.float32)

        metadata = {
            **self.metadata,
            "trace_schema_version": 1,
            "steps": int(steps),
            "chunks": int(chunks),
            "decision_count": decision_count,
            "transition_count": step_count,
            "action_dim": int(predicted_actions.shape[1])
            if predicted_actions.ndim == 2
            else None,
            "index_base": 0,
            "camera_keys": list(_CAMERA_KEYS),
            "state_keys": list(_STATE_KEYS),
        }
        arrays = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "decision_chunk_indices": np.asarray(
                [item["chunk_index"] for item in self.decisions], dtype=np.int64
            ),
            "decision_step_indices": np.asarray(
                [item["step_index"] for item in self.decisions], dtype=np.int64
            ),
            "decision_instructions": np.asarray(
                [item["instruction"] for item in self.decisions], dtype=np.str_
            ),
            "predicted_actions": predicted_actions,
            "predicted_action_offsets": action_offsets,
            "step_indices": np.asarray(
                [item["step_index"] for item in self.steps], dtype=np.int64
            ),
            "step_chunk_indices": np.asarray(
                [item["chunk_index"] for item in self.steps], dtype=np.int64
            ),
            "step_action_indices": np.asarray(
                [item["action_index"] for item in self.steps], dtype=np.int64
            ),
            "executed_actions": _stack([item["action"] for item in self.steps]),
            "step_rewards": np.asarray(
                [item["reward"] for item in self.steps], dtype=np.float32
            ),
            "step_dones": np.asarray(
                [item["done"] for item in self.steps], dtype=np.bool_
            ),
        }
        if any("physical_state_json" in item for item in self.steps):
            arrays["post_physical_state_json"] = np.asarray(
                [item.get("physical_state_json", "{}") for item in self.steps],
                dtype=np.str_,
            )
        if any("contacts_json" in item for item in self.steps):
            arrays["post_contacts_json"] = np.asarray(
                [item.get("contacts_json", "{}") for item in self.steps],
                dtype=np.str_,
            )
        if any("safety_record" in item for item in self.decisions):
            arrays["decision_safety_json"] = np.asarray(
                [
                    json.dumps(
                        item.get("safety_record", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for item in self.decisions
                ],
                dtype=np.str_,
            )
            metadata["safety_decision_schema_version"] = 1
            arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

        for key in _CAMERA_KEYS:
            arrays[f"decision_{key}"] = _stack([item[key] for item in self.decisions])
            arrays[f"post_{key}"] = _stack([item[key] for item in self.steps])
        for key in _STATE_KEYS:
            arrays[f"decision_{key}"] = _stack([item[key] for item in self.decisions])
            arrays[f"post_{key}"] = _stack([item[key] for item in self.steps])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("wb") as output_file:
            np.savez_compressed(output_file, **arrays)
