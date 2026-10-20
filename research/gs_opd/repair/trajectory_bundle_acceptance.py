# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-shot live-grasp gate and Snapshot-only trajectory-bundle acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.trajectory_bundle import (
    BundleConfig,
    generate_trajectory_bundle,
    save_trajectory_bundle,
)
from research.gs_opd.repair.trajectory_tracking import (
    TrackingConfig,
    track_trajectory_bundle,
)
from research.gs_opd.smoke.trajectory_bundle_scene_smoke import (
    _build_local_bypass_scene,
    _check_query,
    _compare_queries,
    _selected_bindings,
)

LIVE_GRASP_REQUIRED_SAMPLES = 3
RANDOM_SELECTION_SMOKE_SEEDS = tuple(range(10))


def select_verified_candidate(tracking: Any, seed: int) -> dict[str, Any] | None:
    """Choose uniformly from candidates that passed real branch validation."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("candidate selection seed must be an integer")
    if (
        not isinstance(tracking, dict)
        or tracking.get("status") != "complete"
        or tracking.get("main_state_restored") is not True
        or tracking.get("snapshot_restore_status") != "verified"
    ):
        return None
    candidate_ids = tracking.get("executable_safe_candidate_ids")
    records = tracking.get("candidates")
    if not isinstance(candidate_ids, list) or not isinstance(records, list):
        return None
    by_id = {
        item.get("candidate_id"): item
        for item in records
        if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
    }
    eligible = []
    for candidate_id in sorted(set(candidate_ids)):
        item = by_id.get(candidate_id)
        compiler = item.get("contact_compiler") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("executable_safe") is not True
            or item.get("physical_safety_verified") is not True
            or item.get("physical_tracking_performed") is not True
            or item.get("physical_tracking_status") != "complete"
            or item.get("main_snapshot_restore_verified") is not True
            or not isinstance(compiler, dict)
            or compiler.get("verdict") != "safe"
        ):
            continue
        eligible.append(item)
    if not eligible:
        return None
    selected = random.Random(seed).choice(eligible)
    candidate = selected.get("geometric_candidate")
    return {
        "seed": seed,
        "candidate_id": selected["candidate_id"],
        "direction": selected.get("direction"),
        "path_length_m": selected.get("path_length_m"),
        "goal_world_m": candidate.get("goal_world_m") if isinstance(candidate, dict) else None,
        "points_world_m": candidate.get("points_world_m") if isinstance(candidate, dict) else None,
        "selection_pool_candidate_ids": [item["candidate_id"] for item in eligible],
        "physical_tracking_status": selected.get("physical_tracking_status"),
        "contact_compiler_verdict": compiler.get("verdict"),
        "main_snapshot_restore_verified": selected.get("main_snapshot_restore_verified"),
    }


def _candidate_route_difference(reference: Any, candidate: Any) -> dict[str, float] | None:
    try:
        paths = [np.asarray(item, dtype=np.float64) for item in (reference, candidate)]
        if any(
            path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all()
            for path in paths
        ):
            return None
        samples = np.linspace(0.0, 1.0, 101)
        normalized = []
        for path in paths:
            lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            distance = np.concatenate(([0.0], np.cumsum(lengths)))
            if distance[-1] <= 1e-12:
                return None
            normalized.append(
                np.column_stack(
                    [np.interp(samples * distance[-1], distance, path[:, axis]) for axis in range(3)]
                )
            )
        delta = np.linalg.norm(normalized[0] - normalized[1], axis=1)
        return {
            "normalized_path_rms_separation_m": float(np.sqrt(np.mean(delta**2))),
            "normalized_path_max_separation_m": float(np.max(delta)),
            "goal_separation_m": float(np.linalg.norm(paths[0][-1] - paths[1][-1])),
        }
    except (TypeError, ValueError):
        return None


def build_candidate_selection_smoke(tracking: Any, primary_seed: int) -> dict[str, Any]:
    """Record one seeded choice and a reproducible ten-seed selection probe."""
    selected = select_verified_candidate(tracking, primary_seed)
    repeat = select_verified_candidate(tracking, primary_seed)
    same_seed_reproducible = selected == repeat if selected is not None else None
    trials = [select_verified_candidate(tracking, seed) for seed in RANDOM_SELECTION_SMOKE_SEEDS]
    valid_trials = [item for item in trials if item is not None]
    reference = valid_trials[0] if valid_trials else None
    reference_points = reference.get("points_world_m") if reference is not None else None
    trial_records = []
    for item in valid_trials:
        item["route_difference_from_first_selection"] = (
            _candidate_route_difference(
                reference_points, item.get("points_world_m")
            )
            if reference is not None
            else None
        )
        item.pop("points_world_m", None)
        trial_records.append(item)
    trial_records_by_seed = {item["seed"]: item for item in trial_records}
    for seed, item in zip(RANDOM_SELECTION_SMOKE_SEEDS, trials):
        if item is None:
            trial_records_by_seed[seed] = {
                "seed": seed,
                "candidate_id": None,
                "selection_status": "no_physically_verified_candidate",
                "route_difference_from_first_selection": None,
            }
    if selected is not None:
        selected.pop("points_world_m", None)
    ids = [
        trial_records_by_seed[seed].get("candidate_id")
        for seed in RANDOM_SELECTION_SMOKE_SEEDS
    ]
    return {
        "policy": "uniform random among physically verified candidates; sorted candidate IDs stabilize the pool order",
        "primary_seed": primary_seed,
        "primary_selection": selected,
        "same_seed_reproducible": same_seed_reproducible,
        "selection_pool_candidate_ids": (
            selected["selection_pool_candidate_ids"] if selected is not None else []
        ),
        "ten_seed_trials": [
            trial_records_by_seed[seed] for seed in RANDOM_SELECTION_SMOKE_SEEDS
        ],
        "ten_seed_unique_candidate_count": len(
            {item for item in ids if item is not None}
        ),
        "ten_seed_selected_candidate_ids": ids,
        "different_verified_route_selected": len(set(ids)) > 1,
        "main_environment_execution": False,
        "note": "all budget-qualified candidates were physically tracked in isolated OSC Snapshot branches; seed trials test pool selection only",
    }


def must_stop_rollout_after_acceptance(report: Any) -> bool:
    """Do not issue a Student action after a failed caller-state integrity check."""
    return bool(
        isinstance(report, dict)
        and (
            report.get("main_state_restored") is False
            or report.get("status")
            in ("snapshot_restore_failed", "state_consistency_failed")
        )
    )


def classify_opposing_finger_contacts(
    contacts: Any,
    *,
    object_geom_ids: set[int],
    object_body_ids: set[int],
    gripper_geom_groups: dict[str, set[int]],
) -> dict[str, Any]:
    """Match contact records to active model-classified left/right finger geoms."""
    if not isinstance(contacts, dict) or contacts.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "left": False,
            "right": False,
            "contact_count": 0,
            "matched_contacts": {"left": [], "right": []},
        }
    records = contacts.get("contacts")
    if not isinstance(records, list):
        return {
            "status": "unavailable",
            "left": False,
            "right": False,
            "contact_count": 0,
            "matched_contacts": {"left": [], "right": []},
        }
    matches: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    try:
        for contact in records:
            geoms = [contact["geom1"], contact["geom2"]]
            bodies = [contact["body1"], contact["body2"]]
            geom_ids = [int(item["id"]) for item in geoms]
            body_ids = [int(item["id"]) for item in bodies]
            for side in ("left", "right"):
                side_geoms = gripper_geom_groups.get(side)
                if not isinstance(side_geoms, set) or not side_geoms:
                    raise ValueError(f"active_{side}_finger_geom_group_unavailable")
                for index in (0, 1):
                    other = 1 - index
                    if (
                        geom_ids[index] in side_geoms
                        and geom_ids[other] in object_geom_ids
                        and body_ids[other] in object_body_ids
                    ):
                        matches[side].append(
                            {
                                "finger_geom": geoms[index],
                                "object_geom": geoms[other],
                                "object_body": bodies[other],
                                "position_world_m": contact.get("position"),
                            }
                        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "left": False,
            "right": False,
            "contact_count": 0,
            "matched_contacts": matches,
            "error": f"{type(exc).__name__}: {exc}",
        }
    left, right = bool(matches["left"]), bool(matches["right"])
    return {
        "status": "pass" if left and right else "fail",
        "left": left,
        "right": right,
        "contact_count": len(matches["left"]) + len(matches["right"]),
        "matched_contacts": matches,
    }


class GraspStabilityMonitor:
    """Require bilateral contact and closed-gripper evidence at distinct states."""

    def __init__(self, required_samples: int = LIVE_GRASP_REQUIRED_SAMPLES):
        if (
            isinstance(required_samples, bool)
            or not isinstance(required_samples, int)
            or required_samples < 2
        ):
            raise ValueError("required_samples must be an integer >= 2")
        self.required_samples = required_samples
        self.consecutive_samples = 0
        self.last_state_id: str | None = None
        self.samples: list[dict[str, Any]] = []
        self.unavailable_count = 0

    @property
    def stable(self) -> bool:
        return self.consecutive_samples >= self.required_samples

    def update(self, response: Any, *, rollout_step: int) -> dict[str, Any]:
        valid_record = isinstance(response, dict)
        finger = response.get("finger_contacts") if valid_record else None
        state_id = response.get("state_id") if valid_record else None
        evidence_pass = bool(
            valid_record
            and response.get("status") == "pass"
            and response.get("grasp_confirmed") is True
            and response.get("gripper_closed") is True
            and isinstance(finger, dict)
            and finger.get("status") == "pass"
            and finger.get("left") is True
            and finger.get("right") is True
            and response.get("physical_state_unchanged") is True
            and isinstance(state_id, str)
            and state_id
        )
        duplicate_state = isinstance(state_id, str) and state_id == self.last_state_id
        accepted = evidence_pass and not duplicate_state
        if not valid_record or response.get("status") == "unavailable":
            self.unavailable_count += 1
        if duplicate_state:
            # Re-reading an unchanged state cannot establish temporal stability.
            if not evidence_pass:
                self.consecutive_samples = 0
        else:
            if isinstance(state_id, str) and state_id:
                self.last_state_id = state_id
            self.consecutive_samples = self.consecutive_samples + 1 if accepted else 0
        sample = {
            "rollout_step": int(rollout_step),
            "state_id": state_id,
            "status": response.get("status") if valid_record else "unavailable",
            "grasp_confirmed": response.get("grasp_confirmed") is True
            if valid_record
            else False,
            "gripper_closed": response.get("gripper_closed") is True
            if valid_record
            else False,
            "finger_contacts": finger,
            "physical_state_unchanged": response.get("physical_state_unchanged") is True
            if valid_record
            else False,
            "reason": response.get("reason", response.get("error"))
            if valid_record
            else "malformed_grasp_state_response",
            "duplicate_state": duplicate_state,
            "counted_as_stable_sample": accepted,
            "consecutive_stable_samples": self.consecutive_samples,
        }
        self.samples.append(sample)
        return sample

    def diagnostics(self) -> dict[str, Any]:
        return {
            "required_distinct_consecutive_states": self.required_samples,
            "current_consecutive_states": self.consecutive_samples,
            "stable_grasp_observed": self.stable,
            "unique_state_samples": sum(
                not item["duplicate_state"] for item in self.samples
            ),
            "valid_grasp_samples": sum(
                item["counted_as_stable_sample"] for item in self.samples
            ),
            "unavailable_sample_count": self.unavailable_count,
            "samples": self.samples,
        }


def prepare_live_acceptance_context(
    connection: Any,
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    instruction: str,
    *,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Validate the fixed task's complete Teacher bindings against live model IDs."""
    inventory = _bridge_request(connection, {"op": "geometry_inventory"})
    payload, validation = _selected_bindings(
        teacher, semantic_results, inventory, subject_id, instruction
    )
    return {
        "teacher": teacher,
        "semantic_results": semantic_results,
        "instruction": instruction,
        "subject_id": payload["carried_binding"]["object_id"],
        "query_payload": payload,
        "binding_validation": validation,
        "initial_geometry_inventory": inventory,
    }


def query_live_grasp(connection: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Ask the existing Student Bridge for read-only current finger contact evidence."""
    return _bridge_request(
        connection,
        {
            "op": "grasp_state",
            "carried_binding": context["query_payload"]["carried_binding"],
            "expected_instruction": context["instruction"],
        },
    )


def _bridge_request(connection: Any, request: dict[str, Any]) -> dict[str, Any]:
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Safety Bridge returned a non-object response for {request.get('op')}"
        )
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response


def _diagnostic_equal(left: Any, right: Any, atol: float = 1e-6) -> dict[str, Any]:
    mismatches: list[str] = []
    maximum_error = 0.0

    def compare(a: Any, b: Any, path: str) -> None:
        nonlocal maximum_error
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                mismatches.append(f"{path}:keys")
                return
            for key in a:
                compare(a[key], b[key], f"{path}.{key}")
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatches.append(f"{path}:length")
                return
            for index, (av, bv) in enumerate(zip(a, b)):
                compare(av, bv, f"{path}[{index}]")
            return
        if isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                mismatches.append(path)
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isfinite(float(a)) or not math.isfinite(float(b)):
                mismatches.append(f"{path}:nonfinite")
                return
            error = abs(float(a) - float(b))
            maximum_error = max(maximum_error, error)
            if error > atol:
                mismatches.append(path)
            return
        if a != b:
            mismatches.append(path)

    compare(left, right, "snapshot_state")
    return {
        "matched": not mismatches,
        "atol": atol,
        "maximum_numeric_error": maximum_error,
        "mismatches": mismatches[:64],
        "mismatch_count": len(mismatches),
    }


def _integration_matches_snapshot(
    snapshot_state: Any, geometry_query: Any
) -> dict[str, Any]:
    expected = np.asarray(
        snapshot_state.get("mujoco_integration_state")
        if isinstance(snapshot_state, dict)
        else None,
        dtype=np.float64,
    )
    actual = np.asarray(
        geometry_query.get("mujoco_integration_state")
        if isinstance(geometry_query, dict)
        else None,
        dtype=np.float64,
    )
    matched = bool(
        expected.ndim == actual.ndim == 1
        and expected.size > 0
        and expected.shape == actual.shape
        and np.isfinite(expected).all()
        and np.isfinite(actual).all()
        and np.array_equal(expected, actual)
    )
    return {
        "matched": matched,
        "snapshot_shape": list(expected.shape),
        "geometry_shape": list(actual.shape),
        "maximum_absolute_error": (
            float(np.max(np.abs(expected - actual)))
            if expected.shape == actual.shape and expected.size
            else None
        ),
        "comparison": "exact finite full mujoco_integration_state equality",
    }


def run_snapshot_trajectory_acceptance(
    connection: Any,
    context: dict[str, Any],
    trigger_grasp_state: dict[str, Any],
    output_json: str | Path,
    *,
    decision_step: int,
    grasp_stability: dict[str, Any],
    candidate_selection_seed: int = 0,
    tracking_config: TrackingConfig = TrackingConfig(),
) -> dict[str, Any]:
    """Generate and track the complete live geometry bundle in isolated branches."""
    output_path = Path(output_json)
    if output_path.exists():
        raise FileExistsError(
            f"Choose a fresh trajectory validation report path: {output_path}"
        )
    bundle_dir = output_path.parent / f"{output_path.stem}_bundle"
    report: dict[str, Any] = {
        "operation": "one_shot_geometry_guided_trajectory_physical_acceptance",
        "status": "blocked",
        "decision_step": int(decision_step),
        "grasp_stability": grasp_stability,
        "main_environment_actions_by_acceptance": 0,
        "main_environment_candidate_execution": False,
        "training_label_created": False,
        "physical_safety_claim": False,
        "trigger_grasp_state": trigger_grasp_state,
        "binding_validation": context.get("binding_validation"),
        "binding_source": "current supplied Safety IR v2 and semantic mapping checked against active Bridge model",
        "current_visual_reidentification_performed": False,
        "candidate_count": 0,
        "geometric_screen_pass_count": None,
        "branch_candidate_count": 0,
        "candidates": [],
        "snapshot": None,
        "restore": {"attempted": False, "verified": False},
        "failure_reasons": [],
    }
    snapshot_id = None
    snapshot_state = None
    geometry_before = None
    try:
        if (
            not isinstance(grasp_stability, dict)
            or grasp_stability.get("stable_grasp_observed") is not True
            or grasp_stability.get("current_consecutive_states", 0)
            < grasp_stability.get(
                "required_distinct_consecutive_states", LIVE_GRASP_REQUIRED_SAMPLES
            )
            or not isinstance(trigger_grasp_state, dict)
            or trigger_grasp_state.get("counted_as_stable_sample") is not True
        ):
            raise ValueError("live_stable_grasp_evidence_precondition_not_met")
        snapshot = _bridge_request(connection, {"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        snapshot_state = snapshot.get("snapshot_state")
        report["snapshot"] = {
            "snapshot_id": snapshot_id,
            "captured": isinstance(snapshot_id, str) and bool(snapshot_id),
            "snapshot_state": snapshot_state,
        }
        if (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or not isinstance(snapshot_state, dict)
        ):
            raise ValueError("snapshot_capture_missing_id_or_canonical_state")

        teacher_path = context.get("teacher_json_path")
        semantic_path = context.get("semantic_results_json_path")
        if teacher_path is not None:
            from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

            teacher_text = Path(teacher_path).read_text(encoding="utf-8")
            context = {
                **context,
                "teacher": parse_teacher_output_v2(teacher_text),
            }
            report["teacher_source"] = {
                "path": str(Path(teacher_path).resolve()),
                "sha256": hashlib.sha256(teacher_text.encode("utf-8")).hexdigest(),
                "safety_ir_v2": context["teacher"],
            }
        if semantic_path is not None:
            semantic_text = Path(semantic_path).read_text(encoding="utf-8")
            semantic_results = json.loads(semantic_text)
            if not isinstance(semantic_results, dict):
                raise ValueError("semantic_results_json_must_contain_an_object")
            context = {**context, "semantic_results": semantic_results}
            report["semantic_binding_source"] = {
                "path": str(Path(semantic_path).resolve()),
                "sha256": hashlib.sha256(semantic_text.encode("utf-8")).hexdigest(),
                "semantic_bindings": semantic_results,
            }

        inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        query_payload, binding_validation = _selected_bindings(
            context["teacher"],
            context["semantic_results"],
            inventory,
            context.get("subject_id"),
            context["instruction"],
        )
        report["binding_validation_at_snapshot"] = binding_validation
        if binding_validation.get("valid") is not True:
            raise ValueError(
                f"snapshot_binding_validation_failed:{binding_validation.get('errors')}"
            )
        report["geometry_inventory"] = inventory
        report["query_payload"] = query_payload

        live_grasp = _bridge_request(
            connection,
            {
                "op": "grasp_state",
                "carried_binding": query_payload["carried_binding"],
                "expected_instruction": context["instruction"],
            },
        )
        report["live_grasp_at_snapshot"] = live_grasp
        if (
            live_grasp.get("status") != "pass"
            or live_grasp.get("grasp_confirmed") is not True
        ):
            raise ValueError(
                f"snapshot_live_grasp_not_confirmed:{live_grasp.get('reason', live_grasp.get('error'))}"
            )

        geometry_before = _bridge_request(connection, query_payload)
        report["planning_geometry"] = geometry_before
        report["same_state_binding_validation"] = _check_query(
            geometry_before,
            context["teacher"],
            context["semantic_results"],
            query_payload,
        )
        state_match = _integration_matches_snapshot(snapshot_state, geometry_before)
        report["snapshot_geometry_state_consistency"] = state_match
        if not state_match["matched"]:
            raise ValueError(
                "planning_geometry_does_not_match_captured_snapshot_integration_state"
            )
        if geometry_before.get("state_id") != live_grasp.get("state_id"):
            raise ValueError("live_grasp_and_planning_geometry_state_id_mismatch")

        repair_context = _bridge_request(connection, {"op": "repair_context"})
        gripper_inventory = _bridge_request(
            connection, {"op": "gripper_geometry_inventory"}
        )
        report["active_osc_configuration"] = repair_context
        report["gripper_geometry_inventory"] = gripper_inventory
        if repair_context.get("status") != "available":
            raise ValueError(
                f"active_osc_configuration_unavailable:{repair_context.get('error')}"
            )
        if gripper_inventory.get("status") != "complete":
            raise ValueError(
                f"gripper_geometry_inventory_incomplete:{gripper_inventory.get('error')}"
            )

        scene = _build_local_bypass_scene(geometry_before, query_payload)
        report["scene"] = scene
        bundle = generate_trajectory_bundle(scene, BundleConfig(candidate_count=18))
        report["trajectory_bundle"] = bundle
        report["candidate_count"] = bundle.get(
            "candidate_count", len(bundle.get("candidates", []))
        )
        report["geometric_screen_pass_count"] = bundle.get("screen_pass_count")
        bundle_outputs = save_trajectory_bundle(bundle, bundle_dir)
        report["bundle_outputs"] = bundle_outputs
        carried_binding = geometry_before.get("objects", {}).get(context["subject_id"])
        if not isinstance(carried_binding, dict):
            raise ValueError("same_state_carried_object_geometry_missing")

        tracking = track_trajectory_bundle(
            connection=connection,
            snapshot_id=snapshot_id,
            bundle=bundle,
            carried_binding=carried_binding,
            repair_context=repair_context,
            teacher=context["teacher"],
            semantic_results=context["semantic_results"],
            geometry_inventory=geometry_before["geometry_inventory"],
            gripper_inventory=gripper_inventory,
            config=tracking_config,
        )
        report["tracking"] = tracking
        report["candidate_selection"] = build_candidate_selection_smoke(
            tracking, candidate_selection_seed
        )
        report["branch_candidate_count"] = tracking.get("tracked_candidate_count", 0)
        report["candidates"] = tracking.get("candidates", [])
        passed_ids = tracking.get("executable_safe_candidate_ids", [])
        if tracking.get("status") == "complete":
            if passed_ids:
                report["status"] = "complete_with_verified_tracking_candidate"
                report["candidate_outcome"] = "one_or_more_candidates_passed_all_checks"
            else:
                report["status"] = "complete_no_candidate_passed"
                report["candidate_outcome"] = "all_tracked_candidates_rejected"
                report["failure_reasons"].append(
                    "no_candidate_passed_complete_tracking_contact_compiler_and_task_segment_checks"
                )
        elif tracking.get("status") == "no_geometric_candidates":
            report["status"] = "no_geometric_candidate_passed_screen"
            report["candidate_outcome"] = (
                "bundle_candidates_rejected_by_geometry_screen"
            )
            report["failure_reasons"].append(
                "no_candidate_passed_conservative_continuous_geometry_screen"
            )
        else:
            report["status"] = "tracking_incomplete"
            report["candidate_outcome"] = "branch_tracking_or_restore_incomplete"
        report["executable_safe_candidate_ids"] = passed_ids
        report["actual_executed_branch_steps"] = sum(
            int(item.get("executed_steps", 0))
            for item in report["candidates"]
            if isinstance(item, dict) and isinstance(item.get("executed_steps", 0), int)
        )
        rejection_counts: dict[str, int] = {}
        for item in report["candidates"]:
            if not isinstance(item, dict):
                continue
            for reason in item.get("rejection_reasons", []):
                if isinstance(reason, str):
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        report["candidate_rejection_reason_counts"] = rejection_counts
        if tracking.get("status") not in ("complete", "no_geometric_candidates"):
            report["failure_reasons"].extend(tracking.get("response_errors", []))
            if not tracking.get("response_errors"):
                report["failure_reasons"].append(
                    f"tracking_status_{tracking.get('status', 'missing')}"
                )
    except Exception as exc:
        report["status"] = "blocked"
        report["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
    finally:
        if snapshot_id is not None:
            report["restore"]["attempted"] = True
            try:
                restored = _bridge_request(
                    connection, {"op": "restore", "snapshot_id": snapshot_id}
                )
                restored_state = restored.get("snapshot_state")
                comparison = _diagnostic_equal(snapshot_state, restored_state)
                report["restore"].update(
                    {
                        "response_received": True,
                        "restored_snapshot_id": restored.get("restored_snapshot_id"),
                        "diagnostic_comparison": comparison,
                        "verified": (
                            restored.get("restored_snapshot_id") == snapshot_id
                            and comparison.get("matched") is True
                        ),
                    }
                )
                if report["restore"]["verified"] is not True:
                    report["failure_reasons"].append(
                        "final_snapshot_restore_diagnostic_mismatch"
                    )
            except Exception as exc:
                report["restore"].update(
                    {
                        "response_received": False,
                        "error": f"{type(exc).__name__}:{exc}",
                        "verified": False,
                    }
                )
                report["failure_reasons"].append("final_snapshot_restore_failed")

        if geometry_before is not None and report["restore"].get("verified") is True:
            try:
                query_payload = report["query_payload"]
                geometry_after = _bridge_request(connection, query_payload)
                _check_query(
                    geometry_after,
                    context["teacher"],
                    context["semantic_results"],
                    query_payload,
                )
                comparison = _compare_queries(geometry_before, geometry_after)
                report["planning_geometry_after_restore"] = geometry_after
                report["geometry_after_restore_comparison"] = comparison
                if comparison.get("matched") is not True:
                    report["failure_reasons"].append(
                        "planning_geometry_changed_after_snapshot_restore"
                    )
            except Exception as exc:
                report["geometry_after_restore_error"] = f"{type(exc).__name__}:{exc}"
                report["status"] = "state_consistency_failed"
                report["failure_reasons"].append(
                    "post_restore_geometry_query_incomplete"
                )

    report["main_state_restored"] = (
        report["restore"].get("verified")
        if report["restore"].get("attempted") is True
        else None
    )
    report["final_task_success_confirmed"] = False
    report["training_label_created"] = False
    report["physical_safety_claim"] = False
    if (
        report["restore"].get("attempted") is True
        and report["restore"].get("verified") is not True
    ):
        report["status"] = "snapshot_restore_failed"
    elif report.get("geometry_after_restore_comparison", {}).get("matched") is False:
        report["status"] = "state_consistency_failed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
