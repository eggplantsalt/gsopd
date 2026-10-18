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
"""Geometry-guided local escape, physical verification, and episode recovery.

Direct recovery commands use absolute controller/workspace/slew constraints.
Relative-to-Student correction budgets apply only to the legacy bias baselines.
Planning never advances episode state: only observed main-environment execution
can commit a side, clearance history, or recovery transition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from research.gs_opd.repair.near_far import (
    GAP_THRESHOLDS_M,
    action_chunk_fingerprint,
    forecast_near_far_risk,
)
from research.gs_opd.repair.verifier_guided_repair import (
    _candidate_constraints,
    _controller_fields,
    _unit,
    build_repair_actions,
)
from research.gs_opd.smoke import early_repair_s70_smoke as probe
from research.gs_opd.smoke.contact_safety_compiler_smoke import (
    _complete_monitoring,
    compile_forbid_contact,
)


@dataclass(frozen=True)
class RecoveryConfig:
    max_candidates: int = 36
    grades_m: tuple[float, ...] = (0.008, 0.016, 0.024)
    maximum_command_slew_m: float = 0.05
    minimum_lateral_progress_m: float = 0.002
    maximum_gap_loss_m: float = 0.005
    maximum_recovery_chunks: int = 12
    maximum_episode_recovery_chunks: int = 24
    maximum_recovery_entries: int = 4
    maximum_retreat_chunks: int = 2
    maximum_stationary_chunks: int = 1
    maximum_stagnant_chunks: int = 3
    clear_chunks_required: int = 2


@dataclass
class RecoveryState:
    phase: str = "normal"
    side: int = 0
    lateral_axis_world: list[float] | None = None
    recovery_chunks: int = 0
    total_recovery_chunks: int = 0
    recovery_entries: int = 0
    retreat_chunks: int = 0
    stationary_chunks: int = 0
    stagnant_chunks: int = 0
    clear_chunks: int = 0
    last_actual_action: list[float] | None = None
    last_actual_gap_m: float | None = None
    actual_gap_history_m: list[float] = field(default_factory=list)
    instruction: str | None = None
    binding_pair: list[int] | None = None


def _array(actions: Any) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if values.shape != (5, 7) or not np.isfinite(values).all():
        raise ValueError("Recovery requires a finite five-step (5,7) chunk")
    return values.astype(np.float64)


def _world_commands(actions: np.ndarray, controller: dict) -> np.ndarray:
    mid_in = (controller["input_min"] + controller["input_max"]) / 2
    mid_out = (controller["output_min"] + controller["output_max"]) / 2
    return (actions[:, :3] - mid_in) * controller["gain_m_per_env_action"] + mid_out


def command_constraints(
    actions: Any,
    context: dict[str, Any],
    previous_action: Any = None,
    config: RecoveryConfig = RecoveryConfig(),
) -> dict[str, Any]:
    """Check final OSC commands and absolute command continuity, without clipping."""
    controller, error = _controller_fields(context)
    if controller is None:
        return {"status": "unavailable", "reason": error}
    if controller["fields"].get("input_ref_frame") not in (None, "world"):
        return {
            "status": "unavailable",
            "reason": "Recovery requires world-frame OSC commands",
        }
    try:
        values = _array(actions)
        bounds = probe._osc_input_bounds_check(values, context)
        world = _world_commands(values, controller)
        adjacent = np.linalg.norm(np.diff(world, axis=0), axis=1)
        boundary = None
        if previous_action is not None:
            previous = np.asarray(previous_action, dtype=np.float64)
            if previous.shape != (7,) or not np.isfinite(previous).all():
                raise ValueError("previous real action is invalid")
            boundary = float(
                np.linalg.norm(
                    world[0] - _world_commands(previous[None, :], controller)[0]
                )
            )
        maximum = max(float(adjacent.max()), boundary or 0.0)
        reasons = []
        if bounds.get("status") != "pass":
            reasons.append("recovery_final_OSC_input_out_of_bounds")
        if maximum > config.maximum_command_slew_m + 1e-8:
            reasons.append("recovery_absolute_command_slew_exceeded")
        return {
            "status": "fail" if reasons else "pass",
            "reason": reasons[0] if reasons else None,
            "rejection_reasons": reasons,
            "input_bounds": bounds,
            "world_commands_m": world.tolist(),
            "maximum_adjacent_command_change_m": float(adjacent.max()),
            "boundary_command_change_m": boundary,
            "boundary_source": "last_actual_main_action"
            if previous_action is not None
            else "first_episode_action_no_previous_command",
            "maximum_allowed_command_change_m": config.maximum_command_slew_m,
            "input_clipping_applied": False,
        }
    except (ValueError, TypeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def generate_recovery_candidates(
    a0: Any,
    directions: list[dict[str, Any]],
    context: dict[str, Any],
    state: RecoveryState,
    config: RecoveryConfig = RecoveryConfig(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate direct commands plus six representative legacy bias baselines."""
    original = _array(a0)
    if not 1 <= config.max_candidates <= 36:
        raise ValueError("Recovery candidate budget must be in 1..36")
    controller, error = _controller_fields(context)
    if controller is None:
        raise ValueError(error)
    if probe._osc_input_bounds_check(original, context).get("status") != "pass":
        raise ValueError("Student A0 is outside the active OSC input bounds")
    by_name = {item["name"]: item["direction_world"] for item in directions}
    away = _unit(by_name.get("nearest_geom_away"), "away")
    lateral = _unit(by_name.get("nearest_geom_lateral_positive"), "lateral")
    if away is None or lateral is None:
        raise ValueError("validated nearest-Geom directions are missing")
    if state.lateral_axis_world is not None:
        prior = _unit(state.lateral_axis_world, "episode lateral axis")
        if prior is None:
            raise ValueError("invalid episode recovery axis")
        lateral = _unit(prior - np.dot(prior, away) * away, "consistent tangent")
        if lateral is None or np.dot(lateral, prior) < 0.1:
            raise ValueError(
                "geometry no longer supports the established bypass direction"
            )
    up = controller["world_up"]
    world_a0 = _world_commands(original, controller)
    previous = (
        _world_commands(np.asarray(state.last_actual_action)[None, :], controller)[0]
        if state.last_actual_action is not None
        else np.zeros(3)
    )
    # Blend from the last real command; subsequent chunks inherit the actual
    # final command rather than restarting from Student's approaching A0.
    blend = np.asarray([0.5, 1.0, 1.0, 1.0, 1.0])[:, None]
    targets = [
        ("stop_approach", 0, world_a0 - np.minimum(world_a0 @ away, 0)[:, None] * away),
        ("hold", 0, np.zeros((5, 3))),
    ]
    for grade in config.grades_m:
        if not np.isfinite(grade) or grade <= 0:
            raise ValueError("Recovery displacement grades must be positive and finite")
        for mode, side, direction in (
            ("lateral", 1, lateral),
            ("lateral", -1, -lateral),
            ("lift", 0, up),
            ("lateral_lift", 1, _unit(lateral + up, "lateral up")),
            ("lateral_lift", -1, _unit(-lateral + up, "lateral up")),
            ("retreat", 0, away),
        ):
            if direction is None:
                continue
            targets.append((mode, side, np.tile(grade * direction, (5, 1))))
    specs, rejected = [], []
    for mode, side, target in targets[: config.max_candidates]:
        world = previous + blend * (target - previous)
        candidate = original.copy()
        mid_in = (controller["input_min"] + controller["input_max"]) / 2
        mid_out = (controller["output_min"] + controller["output_max"]) / 2
        candidate[:, :3] = (world - mid_out) / controller[
            "gain_m_per_env_action"
        ] + mid_in
        candidate = candidate.astype(np.float32).astype(np.float64)
        actual_world = _world_commands(candidate, controller)
        bias = actual_world - world_a0
        diagnostics = {
            "candidate_type": mode,
            "validation_policy": "recovery_absolute_control",
            "side": side,
            "lateral_axis_world": lateral.tolist(),
            "away_world": away.tolist(),
            "world_commands_m": actual_world.tolist(),
            "physical_xyz_bias_by_step_m": bias.tolist(),
            "relative_A0_maximum_correction_m": float(
                np.linalg.norm(bias, axis=1).max()
            ),
            "relative_A0_total_correction_m": float(np.linalg.norm(bias, axis=1).sum()),
            "relative_A0_budget_role": "diagnostic_only_for_direct_recovery",
            "only_xyz_changed": bool(np.array_equal(candidate[:, 3:], original[:, 3:])),
        }
        checked = command_constraints(
            candidate, context, state.last_actual_action, config
        )
        item = {
            "candidate_id": f"recovery-{len(specs) + len(rejected) + 1:03d}",
            "actions": candidate.tolist(),
            "diagnostics": diagnostics,
            "command_constraints": checked,
        }
        (specs if checked["status"] == "pass" else rejected).append(item)
    # Keep the old builder and its original relative-correction limits for
    # comparative baselines; they do not define executable recovery bounds.
    for direction in directions:
        for anchor, (lead, tail) in zip((1, 2), probe.WINDOWS):
            if len(specs) + len(rejected) >= config.max_candidates:
                break
            actions, diag = build_repair_actions(
                original,
                direction["direction_world"],
                probe.AMPLITUDES_M[0],
                anchor,
                lead,
                tail,
                context,
                probe.PROBE_CONFIG,
            )
            item = {
                "candidate_id": f"recovery-baseline-{len(specs) + len(rejected) + 1:03d}",
                "diagnostics": {
                    **diag,
                    "candidate_type": "legacy_bias",
                    "validation_policy": "legacy_relative_A0",
                    "side": 0,
                    "lateral_axis_world": lateral.tolist(),
                },
            }
            if actions is None or diag.get("valid") is not True:
                item["command_constraints"] = {
                    "status": "fail",
                    "reason": diag.get("reason"),
                }
                rejected.append(item)
                continue
            values = _array(actions)
            item["actions"] = values.tolist()
            checked = command_constraints(
                values, context, state.last_actual_action, config
            )
            item["command_constraints"] = checked
            bias = (values[:, :3] - original[:, :3]) * controller[
                "gain_m_per_env_action"
            ]
            item["diagnostics"].update({
                "physical_xyz_bias_by_step_m": bias.tolist(),
                "relative_A0_maximum_correction_m": float(
                    np.linalg.norm(bias, axis=1).max()
                ),
                "relative_A0_total_correction_m": float(
                    np.linalg.norm(bias, axis=1).sum()
                ),
            })
            (specs if checked["status"] == "pass" else rejected).append(item)
    return specs, rejected


def _workspace_constraints(actions, candidate, controller):
    """Check true affine OSC goals, observed EEF positions and model joint limits."""
    try:
        eef = np.asarray(
            [
                candidate["start_state"]["robot_states"][0]["eef_pos"],
                *[
                    step["state"]["robot_states"][0]["eef_pos"]
                    for step in candidate["steps"]
                ],
            ],
            dtype=np.float64,
        )
        if eef.shape != (6, 3) or not np.isfinite(eef).all():
            raise ValueError("incomplete EEF path")
        if "position_limits" not in controller["fields"]:
            raise ValueError("active controller workspace configuration is unavailable")
        positions = controller["fields"]["position_limits"]
        if positions is not None:
            limits = np.asarray(positions, dtype=np.float64)
            if (
                limits.shape != (2, 3)
                or not np.isfinite(limits).all()
                or np.any(limits[0] > limits[1])
            ):
                raise ValueError("invalid active workspace limits")
            goals = eef[:-1] + _world_commands(_array(actions), controller)
            combined = np.concatenate([eef, goals])
            if np.any(combined < limits[0] - 1e-5) or np.any(
                combined > limits[1] + 1e-5
            ):
                return {"status": "fail", "reason": "recovery_workspace_limit_violated"}
        robot = controller["robot"]
        ranges = np.asarray(robot["joint_ranges"], dtype=np.float64)
        limited = np.asarray(robot["joint_limited"], dtype=bool)
        if (
            ranges.shape != (len(limited), 2)
            or not np.isfinite(ranges).all()
            or np.any(ranges[:, 0] > ranges[:, 1])
        ):
            raise ValueError("invalid joint limits")
        for step in candidate["steps"]:
            qpos = np.asarray(
                step["state"]["robot_states"][0]["joint_qpos"], dtype=np.float64
            )
            if qpos.shape != limited.shape or not np.isfinite(qpos).all():
                raise ValueError("incomplete joint state")
            if np.any(
                limited & ((qpos < ranges[:, 0] - 1e-5) | (qpos > ranges[:, 1] + 1e-5))
            ):
                return {"status": "fail", "reason": "recovery_joint_limit_violated"}
        return {
            "status": "pass",
            "workspace_source": "active_controller_and_MuJoCo_joint_model",
            "cartesian_limits_configured": positions is not None,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "reason": f"recovery_workspace_state_unavailable:{exc}",
        }


def task_continuity(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Check the candidate against its Snapshot, not a collided A0 endpoint."""
    result = {
        "status": "unavailable",
        "evaluation_complete": False,
        "physical_grasp_confirmed": False,
        "task_success_confirmed": False,
        "high_quality_training_label_created": False,
        "basis": "snapshot_relative_object_EEF_kinematic_continuity_only",
    }
    body = binding.get("body_id") if isinstance(binding, dict) else None
    if isinstance(body, bool) or not isinstance(body, int) or body < 0:
        return {**result, "reason": "recovery_task_binding_invalid"}
    path = probe._task_object_eef_path(candidate, body)
    if path[0] is None:
        return {**result, "reason": path[1]}
    base_signals = probe._candidate_reward_and_done(baseline)
    signals = probe._candidate_reward_and_done(candidate)
    if base_signals[0] is None or signals[0] is None:
        return {**result, "reason": "recovery_reward_or_done_unavailable"}
    eef, obj = path
    relative = obj - eef
    maximum = float(np.linalg.norm(relative - relative[0], axis=1).max())
    result.update({
        "evaluation_complete": True,
        "relative_positions_world_m": relative.tolist(),
        "maximum_relative_position_change_m": maximum,
        "object_delta_world_m": (obj[-1] - obj[0]).tolist(),
        "eef_delta_world_m": (eef[-1] - eef[0]).tolist(),
    })
    rewards, dones = signals
    base_rewards, _ = base_signals
    result["candidate_rewards"] = rewards
    result["baseline_rewards"] = base_rewards
    if maximum > probe.MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M:
        return {
            **result,
            "status": "fail",
            "reason": "recovery_object_EEF_relative_position_discontinuity",
        }
    first_done = next((i for i, done in enumerate(dones) if done), None)
    if first_done is not None and sum(rewards[: first_done + 1]) <= 0:
        return {
            **result,
            "status": "fail",
            "reason": "recovery_terminated_without_success_reward",
        }
    lost = (np.cumsum(base_rewards) > 0) & (
        np.cumsum(rewards) + 1e-9 < np.cumsum(base_rewards)
    )
    if lost.any():
        return {
            **result,
            "status": "fail",
            "reason": "recovery_lost_existing_positive_task_reward",
        }
    return {**result, "status": "pass", "reason": None}


def select_recovery_candidate(
    candidates: list[dict[str, Any]],
    state: RecoveryState,
    config: RecoveryConfig = RecoveryConfig(),
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Prefer meaningful same-side bypass, then lift, then bounded fallbacks."""
    eligible, rejected = [], []
    for item in candidates:
        if item.get("accepted") is not True:
            continue
        lateral = item["lateral_progress_signed_m"]
        side = 1 if lateral > 0 else -1 if lateral < 0 else 0
        mode = item["diagnostics"]["candidate_type"]
        requested_side = item["diagnostics"].get("side", 0)
        if state.side and (
            lateral * state.side < -1e-6 or requested_side not in (0, state.side)
        ):
            rejected.append({
                "candidate_id": item["candidate_id"],
                "reason": "opposite_recovery_side",
            })
            continue
        if (
            mode not in ("retreat", "hold", "stop_approach")
            and abs(lateral) >= config.minimum_lateral_progress_m
        ):
            category = 0
        elif (
            mode not in ("retreat", "hold", "stop_approach")
            and item["vertical_progress_m"] >= config.minimum_lateral_progress_m
        ):
            category = 1
        elif mode == "retreat" or (
            mode not in ("hold", "stop_approach")
            and item["away_progress_m"] >= config.minimum_lateral_progress_m
        ):
            if state.retreat_chunks >= config.maximum_retreat_chunks:
                rejected.append({
                    "candidate_id": item["candidate_id"],
                    "reason": "bounded_retreat_budget_exhausted",
                })
                continue
            category = 2
        else:
            if state.stationary_chunks >= config.maximum_stationary_chunks:
                rejected.append({
                    "candidate_id": item["candidate_id"],
                    "reason": "bounded_stationary_budget_exhausted",
                })
                continue
            category = 3
        item = {**item, "selection_category": category, "observed_side": side}
        eligible.append(item)
    eligible.sort(
        key=lambda item: (
            item["selection_category"],
            item["endpoint_gap_m"] < max(GAP_THRESHOLDS_M),
            -min(abs(item["lateral_progress_signed_m"]), 0.02),
            item["actual_world_command_length_m"],
            item["candidate_id"],
        )
    )
    return (eligible[0] if eligible else None), rejected


def _resolve_pair(context):
    from research.gs_opd.smoke.near_far_gap_smoke import (
        _resolve_bound_geom_sets,
        _select_forbid_pair,
    )
    from research.gs_opd.smoke.replay_archive_contact_smoke import (
        _validate_current_bindings,
    )

    teacher, bindings = context["teacher"], context["semantic_results"]
    pairs = {
        (rule["subject_id"], rule["target_id"])
        for rule in teacher.get("object_contact_rules", [])
        if rule.get("predicate") == "contact"
        and rule.get("decision") == "forbid"
        and rule.get("phase") == "any"
    }
    if len(pairs) != 1:
        raise ValueError("Recovery requires one unambiguous bound object-contact pair")
    subject, obstacle = next(iter(pairs))
    _select_forbid_pair(teacher, subject, obstacle)
    checked = _validate_current_bindings(
        teacher, bindings, context["geometry_inventory"]
    )
    if checked.get("valid") is not True:
        raise ValueError("Recovery bindings differ from current Geometry Inventory")
    resolved, target_ids, obstacle_ids = _resolve_bound_geom_sets(
        checked, context["geometry_inventory"], subject, obstacle
    )
    binding = context["task_object_binding"]
    role = next(
        (obj.get("role") for obj in teacher["objects"] if obj.get("id") == subject),
        None,
    )
    if (
        not isinstance(binding, dict)
        or binding.get("object_id") != subject
        or binding.get("body_id") != resolved[subject]["body_id"]
        or binding.get("status") != "matched_by_iou"
        or role != "target_object"
    ):
        raise ValueError("Recovery subject is not the bound task object")
    return subject, obstacle, resolved, target_ids, obstacle_ids


def run_local_escape_and_recovery(
    connection: Any,
    snapshot: dict[str, Any],
    a0: Any,
    baseline: dict[str, Any],
    a0_verdict: str,
    context: dict[str, Any],
    state: RecoveryState,
    instruction: str,
    decision_step: int,
    config: RecoveryConfig = RecoveryConfig(),
) -> dict[str, Any]:
    """Verify one local escape decision and restore the supplied Snapshot.

    Args:
        connection: Existing Safety Bridge connection, owned by the caller.
        snapshot: Current decision Snapshot including its state diagnostics.
        a0: Student's current five world-frame OSC actions, after unnormalization.
        baseline: Full A0 branch with substep contact monitoring and state records.
        a0_verdict: Existing Contact Compiler verdict for the baseline.
        context: Validated rules, model inventories, bindings and active controller.
        state: Actual episode history; planning does not mutate this object.
        instruction: Current observation's task instruction.
        decision_step: Actual episode action count, with no archive dependency.
        config: Candidate and bounded recovery lifecycle policy.

    Returns:
        Decision evidence and exact executable actions, or a fail-closed result.
        Kinematic continuity does not establish a physical grasp or task success.
    """
    report = {
        "status": "local_recovery_unavailable",
        "must_abort_episode": True,
        "main_state_restored": False,
        "training_label_valid": False,
        "training_action_chunk": None,
        "repaired_action_chunk_A_plus": None,
        "repair_attempted": False,
        "failure_reasons": [],
        "recovery_search": {
            "candidate_results": [],
            "generation_rejections": [],
            "verification_complete": False,
            "config": asdict(config),
        },
        "recovery": {
            "state_before": asdict(state),
            "decision_step": decision_step,
            "physical_grasp_confirmed": False,
            "task_success_confirmed": False,
            "high_quality_demonstration": False,
        },
    }
    search = report["recovery_search"]
    selected = None
    try:
        actions = _array(a0)
        if a0_verdict not in ("safe", "violation") or not probe._complete_five_steps(
            baseline
        ):
            raise ValueError("Recovery requires a conclusive full A0 branch")
        _, monitor_error = _complete_monitoring(baseline)
        if monitor_error:
            raise ValueError(monitor_error)
        if (
            probe._compare_mujoco_integration_state(
                snapshot["snapshot_state"], baseline.get("start_state")
            ).get("matched")
            is not True
        ):
            raise ValueError("A0 branch did not start at the decision Snapshot")
        if not np.array_equal(
            np.asarray(
                [step.get("action") for step in baseline["steps"]], dtype=np.float32
            ),
            actions.astype(np.float32),
        ):
            raise ValueError("A0 branch action records differ from Student A0")
        subject, obstacle, resolved, target_ids, obstacle_ids = _resolve_pair(context)
        pair = [resolved[subject]["body_id"], resolved[obstacle]["body_id"]]
        if state.instruction not in (None, instruction) or state.binding_pair not in (
            None,
            pair,
        ):
            raise ValueError("episode instruction or physical binding changed")
        if (
            state.phase == "stopped"
            or state.recovery_chunks >= config.maximum_recovery_chunks
            or state.stagnant_chunks >= config.maximum_stagnant_chunks
        ):
            raise ValueError("bounded_recovery_progress_budget_exhausted")
        forecast = forecast_near_far_risk(
            connection,
            snapshot,
            actions,
            target_ids,
            obstacle_ids,
            decision_step=decision_step,
        )
        report.update({
            "near_far_result": forecast,
            "near_far_status": forecast["status"],
            "near_far_risk_verdict": forecast["risk_verdict"],
        })
        if forecast["status"] != "complete" or forecast["risk_verdict"] not in (
            "warning",
            "no_warning",
        ):
            raise ValueError("recovery_near_far_forecast_incomplete")
        if (
            forecast.get("snapshot_id") != snapshot["snapshot_id"]
            or forecast.get("action_chunk_sha256") != action_chunk_fingerprint(actions)
            or forecast.get("executed_action_steps") != 5
            or len(forecast.get("distance_samples", [])) != 6
            or set(forecast.get("target_geom_ids", [])) != set(target_ids)
            or set(forecast.get("obstacle_geom_ids", [])) != set(obstacle_ids)
            or any(
                forecast.get("snapshot_restore", {}).get(key, {}).get("matched")
                is not True
                for key in ("before", "after")
            )
        ):
            raise ValueError("recovery_near_far_provenance_or_restore_invalid")
        start_gap = float(forecast["distance_samples"][0]["minimum_distance_m"])
        a0_gap = float(forecast["distance_samples"][-1]["minimum_distance_m"])
        if state.last_actual_gap_m is not None and not np.isclose(
            start_gap, state.last_actual_gap_m, atol=1e-6, rtol=0
        ):
            raise ValueError("actual_gap_changed_without_recorded_main_execution")
        report["recovery"].update({
            "start_gap_m": start_gap,
            "a0_endpoint_gap_m": a0_gap,
            "subject_geom_ids": target_ids,
            "obstacle_geom_ids": obstacle_ids,
            "binding_pair": pair,
            "instruction": instruction,
        })
        clear = a0_verdict == "safe" and forecast["risk_verdict"] == "no_warning"
        exit_ready = (
            clear
            and state.clear_chunks >= config.clear_chunks_required
            and start_gap > max(GAP_THRESHOLDS_M)
        )
        active = state.phase != "normal"
        predicted_success = any(float(step["reward"]) > 0 for step in baseline["steps"])
        continuing_clear = (
            clear and active and (a0_gap >= start_gap - 1e-6 or predicted_success)
        )
        if (
            continuing_clear
            and state.lateral_axis_world is not None
            and not predicted_success
        ):
            path = probe._task_object_eef_path(
                baseline, context["task_object_binding"]["body_id"]
            )
            if path[0] is None:
                raise ValueError(
                    "A0 object path missing during recovery clearance confirmation"
                )
            continuing_clear = (
                float(np.dot(path[1][-1] - path[1][0], state.lateral_axis_world))
                * state.side
                >= -1e-6
            )
        if clear and (not active or exit_ready or continuing_clear):
            checked = command_constraints(
                actions, context["repair_context"], state.last_actual_action, config
            )
            if checked["status"] != "pass":
                raise ValueError(checked["reason"])
            controller, error = _controller_fields(context["repair_context"])
            workspace = _workspace_constraints(actions, baseline, controller)
            task_path = probe._task_object_eef_path(
                baseline, context["task_object_binding"]["body_id"]
            )
            if (
                workspace["status"] != "pass"
                or task_path[0] is None
                or probe._candidate_reward_and_done(baseline)[0] is None
            ):
                raise ValueError("unmodified_A0_workspace_or_task_state_incomplete")
            selected = {
                "candidate_id": "verified-A0",
                "actions": actions.tolist(),
                "branch_states": [step["state"] for step in baseline["steps"]],
                "branch_rewards": [step["reward"] for step in baseline["steps"]],
                "branch_dones": [step["done"] for step in baseline["steps"]],
                "endpoint_gap_m": a0_gap,
                "accepted": True,
                "full_five_step_chunk": True,
                "contact_monitoring_complete": True,
                "physical_safety_verdict": "safe",
                "validation_complete": True,
                "command_constraints": checked,
                "workspace_constraints": workspace,
                "diagnostics": {
                    "candidate_type": "student_A0",
                    "side": state.side,
                    "lateral_axis_world": state.lateral_axis_world,
                    "relative_A0_total_correction_m": 0.0,
                },
                "selection_category": None,
                "observed_side": state.side,
            }
            report["recovery"]["transition"] = (
                "exit" if exit_ready else "confirm_clear" if active else "normal"
            )
            search["verification_complete"] = True
        else:
            if (
                state.total_recovery_chunks >= config.maximum_episode_recovery_chunks
                or (
                    not active
                    and state.recovery_entries >= config.maximum_recovery_entries
                )
            ):
                raise ValueError("episode_recovery_budget_exhausted")
            report["repair_attempted"] = True
            endpoint_distance = {
                "query_complete": True,
                "distance_reliable": True,
                "minimum_distance_m": start_gap,
                "minimum_pair": forecast["distance_samples"][0]["minimum_pair"],
            }
            directions, geometry = probe.derive_clearance_directions(
                endpoint_distance, *pair, context["repair_context"]["world_up"]
            )
            search["geometry"] = geometry
            search["geometry"]["recovery_direction_source"] = (
                "current_decision_Snapshot_nearest_pair_not_violated_A0_endpoint"
            )
            specs, rejected = generate_recovery_candidates(
                actions, directions, context["repair_context"], state, config
            )
            search["generation_rejections"] = rejected
            search["generated_candidate_count"] = len(specs)
            if not specs:
                raise ValueError("no_recovery_candidates_passed_command_constraints")
            branch = probe._request(
                connection,
                {
                    "op": "branch_rollout",
                    "snapshot_id": snapshot["snapshot_id"],
                    "candidates": [
                        {
                            "candidate_id": item["candidate_id"],
                            "actions": item["actions"],
                        }
                        for item in specs
                    ],
                },
            )
            if branch.get("main_state_restored") is not True or branch.get("error"):
                raise ValueError("recovery_branch_restore_failed_or_branch_error")
            compiled = compile_forbid_contact(
                context["teacher"],
                context["object_id"],
                context["semantic_result"],
                branch,
                context["geometry_inventory"],
                context["gripper_inventory"],
                object_semantic_results=context["semantic_results"],
            )
            controller, _ = _controller_fields(context["repair_context"])
            incomplete = False
            for spec in specs:
                candidate = probe._branch_candidate(branch, spec["candidate_id"])
                verdict = probe._compiler_candidate_summary(
                    compiled, spec["candidate_id"]
                )
                item = {
                    **spec,
                    "accepted": False,
                    "rejection_reasons": [],
                    "validation_complete": False,
                }
                search["candidate_results"].append(item)
                if not probe._complete_five_steps(candidate):
                    item["rejection_reasons"].append("recovery_branch_chunk_incomplete")
                    incomplete = True
                    continue
                matching_start = probe._compare_mujoco_integration_state(
                    snapshot["snapshot_state"], candidate.get("start_state")
                )
                matching_actions = np.array_equal(
                    np.asarray(
                        [step.get("action") for step in candidate["steps"]],
                        dtype=np.float32,
                    ),
                    np.asarray(spec["actions"], dtype=np.float32),
                )
                if matching_start.get("matched") is not True or not matching_actions:
                    item["rejection_reasons"].append(
                        "recovery_candidate_snapshot_or_action_mismatch"
                    )
                    incomplete = True
                    continue
                _, error = _complete_monitoring(candidate)
                safety = (
                    verdict.get("verdict") if isinstance(verdict, dict) else "unknown"
                )
                item["compiler"] = verdict
                if error or safety not in ("safe", "violation"):
                    item["rejection_reasons"].append(
                        error or "recovery_compiler_unknown"
                    )
                    incomplete = True
                    continue
                workspace = _workspace_constraints(
                    spec["actions"], candidate, controller
                )
                if spec["diagnostics"]["validation_policy"] == "legacy_relative_A0":
                    relative = _candidate_constraints(
                        actions,
                        _array(spec["actions"]),
                        spec["diagnostics"],
                        candidate,
                        controller,
                        probe.PROBE_CONFIG,
                    )
                    item["legacy_relative_constraints"] = relative
                    if relative["status"] != "pass":
                        item["rejection_reasons"].append(relative.get("reason"))
                continuity = task_continuity(
                    candidate, baseline, context["task_object_binding"]
                )
                endpoint = probe.replay_and_measure_endpoint(
                    connection,
                    snapshot["snapshot_id"],
                    snapshot["snapshot_state"],
                    np.asarray(spec["actions"]),
                    candidate,
                    target_ids,
                    obstacle_ids,
                )
                item.update({
                    "workspace_constraints": workspace,
                    "task_continuity": continuity,
                    "endpoint_replay": endpoint,
                    "contact_monitoring_complete": True,
                    "contact_monitoring": probe._monitor_summary(candidate),
                    "full_five_step_chunk": True,
                    "physical_safety_verdict": safety,
                })
                if endpoint.get("main_state_restored") is not True:
                    raise ValueError("recovery_endpoint_snapshot_restore_failed")
                complete = (
                    workspace["status"] in ("pass", "fail")
                    and continuity["evaluation_complete"] is True
                    and endpoint.get("status") == "complete"
                )
                item["validation_complete"] = complete
                incomplete |= not complete
                if safety != "safe":
                    item["rejection_reasons"].append(
                        "forbidden_contact_in_recovery_candidate"
                    )
                for check in (workspace, continuity):
                    if check["status"] != "pass":
                        item["rejection_reasons"].append(check.get("reason"))
                if endpoint.get("status") != "complete":
                    item["rejection_reasons"].append("recovery_endpoint_gap_incomplete")
                gap = endpoint.get("endpoint_distance") or {}
                value = gap.get("minimum_distance_m")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                ):
                    item["rejection_reasons"].append(
                        "recovery_endpoint_gap_unavailable"
                    )
                    incomplete = True
                    continue
                item["endpoint_gap_m"] = float(value)
                floor = max(
                    min(start_gap, max(GAP_THRESHOLDS_M)), min(GAP_THRESHOLDS_M)
                )
                if value < floor or value < start_gap - config.maximum_gap_loss_m:
                    item["rejection_reasons"].append(
                        "recovery_clearance_below_reasonable_margin"
                    )
                if continuity.get("object_delta_world_m") is not None:
                    delta = np.asarray(continuity["object_delta_world_m"])
                    axis = np.asarray(spec["diagnostics"]["lateral_axis_world"])
                    item.update({
                        "lateral_progress_signed_m": float(np.dot(delta, axis)),
                        "vertical_progress_m": float(
                            np.dot(delta, controller["world_up"])
                        ),
                        "away_progress_m": float(
                            np.dot(delta, directions[0]["direction_world"])
                        ),
                        "actual_world_command_length_m": float(
                            np.linalg.norm(
                                _world_commands(_array(spec["actions"]), controller),
                                axis=1,
                            ).sum()
                        ),
                    })
                item["accepted"] = not item["rejection_reasons"] and complete
                if item["accepted"]:
                    item["branch_states"] = [
                        step["state"] for step in candidate["steps"]
                    ]
                    item["branch_rewards"] = [
                        step["reward"] for step in candidate["steps"]
                    ]
                    item["branch_dones"] = [step["done"] for step in candidate["steps"]]
            search["verification_complete"] = not incomplete and len(
                search["candidate_results"]
            ) == len(specs)
            if not search["verification_complete"]:
                raise ValueError("recovery_physical_verification_incomplete")
            selected, exclusions = select_recovery_candidate(
                search["candidate_results"], state, config
            )
            search["selection_rejections"] = exclusions
            search["physically_qualified_candidate_count"] = sum(
                item["accepted"] for item in search["candidate_results"]
            )
            search["selection_eligible_candidate_count"] = search[
                "physically_qualified_candidate_count"
            ] - len(exclusions)
            if selected is None:
                raise ValueError("no_safe_feasible_recovery_candidate")
            for item in search["candidate_results"]:
                if item["candidate_id"] != selected["candidate_id"]:
                    for key in ("branch_states", "branch_rewards", "branch_dones"):
                        item.pop(key, None)
            report["recovery"]["transition"] = "enter" if not active else "continue"
        search["selected_candidate"] = selected
        report.update({
            "status": "a0_safe"
            if selected["candidate_id"] == "verified-A0"
            else "local_recovery_selected",
            "must_abort_episode": False,
            "training_label_valid": True,
            "training_action_chunk": selected["actions"],
            "repaired_action_chunk_A_plus": None
            if selected["candidate_id"] == "verified-A0"
            else selected["actions"],
            "selected_candidate_id": selected["candidate_id"],
            "selected_endpoint_gap_m": selected["endpoint_gap_m"],
            "baseline_a0_endpoint_gap_m": a0_gap,
            "selected_gap_improvement_m": selected["endpoint_gap_m"] - a0_gap,
            "repair_status": "local_recovery_candidate_selected"
            if report["repair_attempted"]
            else "verified_A0",
            "label_source": "verified_safe_local_recovery_chunk"
            if report["repair_attempted"]
            else "compiler_and_near_far_verified_safe_A0",
            "selection_policy": "same_side_lateral_progress_then_lift_then_bounded_retreat",
            "recovery_execution": {
                "snapshot_id": snapshot["snapshot_id"],
                "a0_fingerprint": action_chunk_fingerprint(actions),
                "selected_fingerprint": action_chunk_fingerprint(selected["actions"]),
                "start_state": snapshot["snapshot_state"],
            },
        })
        report["recovery"].update({
            "selected_candidate_type": selected["diagnostics"]["candidate_type"],
            "selected_side": selected.get("observed_side", state.side),
            "selected_endpoint_gap_m": selected["endpoint_gap_m"],
            "selected_lateral_progress_m": selected.get("lateral_progress_signed_m"),
            "relative_A0_total_correction_m": selected["diagnostics"].get(
                "relative_A0_total_correction_m"
            ),
            "relative_A0_maximum_correction_m": selected["diagnostics"].get(
                "relative_A0_maximum_correction_m", 0.0
            ),
            "near_far_warning": forecast["risk_verdict"] == "warning",
        })
    except Exception as exc:
        report["failure_reasons"].append(f"{type(exc).__name__}:{exc}")
    finally:
        try:
            restored = probe._restore_and_compare(
                connection, snapshot["snapshot_id"], snapshot["snapshot_state"]
            )
        except Exception as exc:
            restored = {"matched": False, "error": str(exc)}
        search["final_restore"] = restored
        report["main_state_restored"] = restored.get("matched") is True
        if report["failure_reasons"] or not report["main_state_restored"]:
            report.update({
                "must_abort_episode": True,
                "training_label_valid": False,
                "training_action_chunk": None,
                "repaired_action_chunk_A_plus": None,
            })
            report["status"] = (
                "local_recovery_unavailable"
                if report["main_state_restored"]
                else "aborted_main_state_restore_failed"
            )
        search["rejection_reason_counts"] = dict(
            Counter(
                reason
                for item in search["candidate_results"]
                for reason in item["rejection_reasons"]
            )
        )
        search["constraint_rejection_reason_counts"] = dict(
            Counter(
                item.get("command_constraints", {}).get("reason", "generation_failed")
                for item in search["generation_rejections"]
            )
        )
        search["selection_rejection_reason_counts"] = dict(
            Counter(item["reason"] for item in search.get("selection_rejections", []))
        )
    return report


def select_verified_recovery_execution(
    decision: dict[str, Any],
    predicted: Any,
) -> dict[str, Any]:
    """Bind selected, verified, labeled and executable actions exactly."""
    try:
        if (
            decision.get("must_abort_episode")
            or decision.get("main_state_restored") is not True
        ):
            raise ValueError(
                next(
                    iter(decision.get("failure_reasons", [])),
                    "Recovery restore/execution is unverified",
                )
            )
        a0 = _array(predicted)
        execution = decision["recovery_execution"]
        search = decision["recovery_search"]
        selected = search["selected_candidate"]
        values = _array(selected["actions"])
        if not np.array_equal(values[:, 3:], a0[:, 3:]):
            raise ValueError("Recovery changed Student rotation or gripper actions")
        if selected["candidate_id"] == "verified-A0" and not np.array_equal(values, a0):
            raise ValueError(
                "Unmodified Student A0 selection contains different actions"
            )
        if (
            decision.get("must_abort_episode")
            or not decision.get("training_label_valid")
            or decision.get("main_state_restored") is not True
        ):
            raise ValueError("Recovery did not produce a restored, executable decision")
        if (
            search.get("verification_complete") is not True
            or search.get("final_restore", {}).get("matched") is not True
            or selected.get("accepted") is not True
        ):
            raise ValueError("Recovery physical verification is incomplete")
        if not (
            selected.get("validation_complete") is True
            and selected.get("full_five_step_chunk") is True
            and selected.get("contact_monitoring_complete") is True
            and selected.get("physical_safety_verdict") == "safe"
            and selected.get("command_constraints", {}).get("status") == "pass"
            and selected.get("workspace_constraints", {}).get("status") == "pass"
        ):
            raise ValueError(
                "Selected Recovery lacks full physical/action verification"
            )
        if selected["candidate_id"] != "verified-A0":
            canonical = next(
                (
                    item
                    for item in search["candidate_results"]
                    if item["candidate_id"] == selected["candidate_id"]
                ),
                None,
            )
            if (
                canonical is None
                or canonical.get("accepted") is not True
                or not np.array_equal(_array(canonical["actions"]), values)
                or selected.get("task_continuity", {}).get("status") != "pass"
                or selected.get("endpoint_replay", {}).get("main_state_restored")
                is not True
                or selected.get("endpoint_replay", {}).get("status") != "complete"
            ):
                raise ValueError(
                    "Selected Recovery task/workspace/restore evidence is incomplete"
                )
        if execution["snapshot_id"] != decision["snapshot_id"] or execution[
            "a0_fingerprint"
        ] != action_chunk_fingerprint(a0):
            raise ValueError("Recovery Snapshot or A0 provenance mismatch")
        if execution["selected_fingerprint"] != action_chunk_fingerprint(values):
            raise ValueError("Recovery selected action fingerprint mismatch")
        if not np.array_equal(
            values, _array(decision["training_action_chunk"])
        ) or not np.array_equal(a0, _array(decision["original_action_chunk_A0"])):
            raise ValueError("Recovery training/execution actions differ")
        if selected["candidate_id"] != "verified-A0" and not np.array_equal(
            values, _array(decision["repaired_action_chunk_A_plus"])
        ):
            raise ValueError("Recovery A+ differs from the executable candidate")
        if selected["candidate_id"] != decision["selected_candidate_id"]:
            raise ValueError("Recovery selected candidate ID mismatch")
        return {
            "valid": True,
            "actions": values.astype(np.float32),
            "candidate_id": selected["candidate_id"],
            "source": decision["label_source"],
            "reason": None,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {"valid": False, "reason": str(exc)}


def validate_actual_recovery_step(
    decision: dict[str, Any],
    action_index: int,
    action: Any,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Check each real response before allowing another action from the branch."""
    try:
        selected = decision["recovery_search"]["selected_candidate"]
        if not 0 <= action_index < 5 or not np.array_equal(
            np.asarray(action, dtype=np.float32),
            np.asarray(selected["actions"][action_index], dtype=np.float32),
        ):
            raise ValueError("actual action differs from verified branch action")
        compared = probe._compare_mujoco_integration_state(
            selected["branch_states"][action_index],
            data.get("snapshot_state"),
        )
        if compared.get("matched") is not True:
            return {
                "status": "failed",
                "reason": "actual executed state differs from verified branch",
                "integration_state_check": compared,
            }
        reward, done = data["reward"], data["done"]
        if (
            isinstance(reward, (bool, np.bool_))
            or not np.isfinite(float(reward))
            or not isinstance(done, (bool, np.bool_))
            or not np.isclose(
                float(reward),
                selected["branch_rewards"][action_index],
                atol=1e-9,
                rtol=0,
            )
            or bool(done) != bool(selected["branch_dones"][action_index])
        ):
            raise ValueError("actual reward/done differs from the verified branch")
        return {"status": "complete", "integration_state_check": compared}
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return {"status": "failed", "reason": str(exc)}


def commit_recovery_execution(
    state: RecoveryState,
    decision: dict[str, Any],
    actual_actions: Any,
    post_data: list[dict[str, Any]],
    context: dict[str, Any],
    actual_gap: dict[str, Any],
    config: RecoveryConfig = RecoveryConfig(),
) -> dict[str, Any]:
    """Commit only matched main-environment transitions, never branch predictions."""
    selected = decision["recovery_search"]["selected_candidate"]
    values = np.asarray(actual_actions, dtype=np.float32)
    expected = np.asarray(selected["actions"], dtype=np.float32)
    count = len(values)
    result = {
        "status": "failed",
        "source": "actual_main_environment",
        "executed_steps": count,
        "verified_safe_recovery_chunk": False,
        "high_quality_demonstration": False,
        "task_success_observed": any(
            float(data.get("reward", 0)) > 0 for data in post_data
        ),
    }
    try:
        if (
            not 1 <= count <= 5
            or values.shape != (count, 7)
            or len(post_data) != count
            or not np.array_equal(values, expected[:count])
        ):
            raise ValueError("actual main actions do not match the verified prefix")
        observed = post_data[-1].get("snapshot_state")
        comparison = probe._compare_mujoco_integration_state(
            selected["branch_states"][count - 1], observed
        )
        result["expected_vs_actual_integration_state"] = comparison
        comparisons = [
            probe._compare_mujoco_integration_state(
                selected["branch_states"][index], data.get("snapshot_state")
            )
            for index, data in enumerate(post_data)
        ]
        result["per_step_integration_state_checks"] = comparisons
        if any(check.get("matched") is not True for check in comparisons):
            raise ValueError("actual executed state differs from verified branch")
        step_checks = [
            validate_actual_recovery_step(decision, index, values[index], data)
            for index, data in enumerate(post_data)
        ]
        result["per_step_execution_checks"] = step_checks
        failed = next(
            (check for check in step_checks if check["status"] != "complete"), None
        )
        if failed is not None:
            raise ValueError(failed["reason"])
        if actual_gap.get("distance_available") is not True:
            raise ValueError("actual_endpoint_gap_unavailable")
        gap = float(actual_gap["minimum_distance_m"])
        if isinstance(actual_gap["minimum_distance_m"], bool) or not np.isfinite(gap):
            raise ValueError("actual_endpoint_gap_unavailable")
        if count == 5 and not np.isclose(
            gap, selected["endpoint_gap_m"], atol=1e-6, rtol=0
        ):
            raise ValueError("actual endpoint gap differs from verified branch")
        recovery = decision["recovery"]
        start_state = decision["recovery_execution"]["start_state"]
        body_id = context["task_object_binding"]["body_id"]

        def object_position(data):
            return np.asarray(
                next(
                    obj["root_pos"]
                    for obj in data["object_states"].values()
                    if obj["body_id"] == body_id
                ),
                dtype=np.float64,
            )

        actual_states = [start_state, *[data["snapshot_state"] for data in post_data]]
        relative = np.asarray([
            object_position(data)
            - np.asarray(data["robot_states"][0]["eef_pos"], dtype=np.float64)
            for data in actual_states
        ])
        if relative.shape != (count + 1, 3) or not np.isfinite(relative).all():
            raise ValueError("actual task-object/EEF path is incomplete")
        relative_change = float(np.linalg.norm(relative - relative[0], axis=1).max())
        result["actual_maximum_relative_position_change_m"] = relative_change
        if (
            selected["candidate_id"] != "verified-A0"
            and relative_change > probe.MAX_GRASP_RELATIVE_DISTANCE_CHANGE_M
        ):
            raise ValueError("actual object/EEF continuity failed")
        if post_data[-1].get("done") is True and not result["task_success_observed"]:
            raise ValueError(
                "actual environment terminated without positive success reward"
            )
        if state.instruction not in (
            None,
            recovery["instruction"],
        ) or state.binding_pair not in (None, recovery["binding_pair"]):
            raise ValueError("actual recovery context changed")
        state.instruction = recovery["instruction"]
        state.binding_pair = recovery["binding_pair"]
        state.last_actual_action = values[-1].tolist()
        state.last_actual_gap_m = gap
        state.actual_gap_history_m.append(gap)
        state.actual_gap_history_m = state.actual_gap_history_m[-16:]
        transition = recovery["transition"]
        if transition in ("normal", "exit"):
            state.phase = "normal"
            state.side, state.lateral_axis_world = 0, None
            state.recovery_chunks = state.retreat_chunks = state.stationary_chunks = (
                state.stagnant_chunks
            ) = state.clear_chunks = 0
        else:
            state.phase = (
                "escape" if recovery["near_far_warning"] else "clearance_confirmation"
            )
            state.recovery_chunks += 1
            if transition == "enter":
                state.recovery_entries += 1
            if selected["candidate_id"] != "verified-A0":
                state.total_recovery_chunks += 1
            if state.side == 0 and selected.get("selection_category") == 0:
                state.side = selected["observed_side"]
                state.lateral_axis_world = selected["diagnostics"]["lateral_axis_world"]
            category = selected.get("selection_category")
            state.retreat_chunks += category == 2
            state.stationary_chunks += category == 3
            delta = object_position(observed) - object_position(start_state)
            actual_lateral = (
                float(np.dot(delta, state.lateral_axis_world)) * state.side
                if state.lateral_axis_world is not None
                else 0.0
            )
            result["actual_lateral_progress_m"] = actual_lateral
            controller, error = _controller_fields(context["repair_context"])
            if controller is None:
                raise ValueError(error)
            actual_vertical = float(np.dot(delta, controller["world_up"]))
            result["actual_vertical_progress_m"] = actual_vertical
            progress = (
                actual_lateral >= config.minimum_lateral_progress_m
                or (
                    category == 1
                    and actual_vertical >= config.minimum_lateral_progress_m
                )
                or gap - recovery["start_gap_m"] >= 0.001
                or result["task_success_observed"]
            )
            state.stagnant_chunks = 0 if progress else state.stagnant_chunks + 1
            cleared = (
                not recovery["near_far_warning"]
                and gap > max(GAP_THRESHOLDS_M)
                and gap >= recovery["start_gap_m"] - 1e-6
            )
            state.clear_chunks = state.clear_chunks + 1 if cleared else 0
        result.update({
            "status": "complete",
            "actual_endpoint_gap_m": gap,
            "state_after": asdict(state),
            "verified_safe_recovery_chunk": count == 5
            and selected["candidate_id"] != "verified-A0",
            "must_stop": (
                state.stagnant_chunks >= config.maximum_stagnant_chunks
                or state.recovery_chunks >= config.maximum_recovery_chunks
                or state.total_recovery_chunks >= config.maximum_episode_recovery_chunks
            ),
        })
        if result["must_stop"]:
            result["reason"] = (
                "bounded_recovery_no_progress"
                if state.stagnant_chunks >= config.maximum_stagnant_chunks
                else "bounded_episode_recovery_limit"
                if state.total_recovery_chunks >= config.maximum_episode_recovery_chunks
                else "bounded_recovery_chunk_limit"
            )
            state.phase = "stopped"
            result["state_after"] = asdict(state)
    except (ValueError, KeyError, TypeError, StopIteration) as exc:
        state.phase = "stopped"
        result.update({
            "reason": str(exc),
            "must_stop": True,
            "state_after": asdict(state),
        })
    decision["recovery"]["actual_execution"] = result
    return result


def diagnose_archive_generation(
    archive_path: Any,
    decision_steps: tuple[int, ...] | list[int] = (50, 55),
) -> list[dict[str, Any]]:
    """Use archived commands and recorded controller diagnostics; never simulate."""
    import json

    results = []
    with np.load(archive_path, allow_pickle=False) as archive:
        for step in decision_steps:
            item = {
                "decision_step": int(step),
                "status": "unavailable",
                "physical_safety_verified": False,
            }
            results.append(item)
            try:
                matches = np.flatnonzero(archive["decision_step_indices"] == step)
                if len(matches) != 1:
                    raise ValueError(
                        "archive does not contain a unique requested decision"
                    )
                index = int(matches[0])
                record = json.loads(str(archive["decision_safety_json"][index]))
                offsets = archive["predicted_action_offsets"]
                actions = _array(
                    archive["predicted_actions"][offsets[index] : offsets[index + 1]]
                )
                if not np.array_equal(
                    actions, _array(record["original_action_chunk_A0"])
                ):
                    raise ValueError(
                        "NPZ predicted actions disagree with decision safety record"
                    )
                search = record.get("early_repair_search") or {}
                entries = search.get("candidate_results", []) + search.get(
                    "candidate_generation_rejections", []
                )
                source = next(
                    entry
                    for entry in entries
                    if entry.get("action_diagnostics", {}).get("candidate_family")
                    == "brake_and_steer"
                )
                up_source = next(
                    entry
                    for entry in entries
                    if entry.get("action_diagnostics", {}).get("steering_mode")
                    == "brake_up"
                )
                diag = source["action_diagnostics"]
                gain = np.asarray(diag["control_gain_m_per_env_action"], dtype=float)
                input_mid = np.asarray(diag["control_input_midpoint"], dtype=float)
                output_mid = np.asarray(diag["control_output_midpoint_m"], dtype=float)
                low = np.asarray(source["input_bounds_check"]["input_min"], dtype=float)
                high = np.asarray(
                    source["input_bounds_check"]["input_max"], dtype=float
                )
                up = _unit(
                    up_source["direction"]["steering_direction_world"],
                    "archived world up",
                )
                if (
                    up is None
                    or gain.shape != (3,)
                    or input_mid.shape != (3,)
                    or output_mid.shape != (3,)
                ):
                    raise ValueError("archived controller diagnostics are incomplete")
                context = {
                    "status": "available",
                    "world_up": up.tolist(),
                    "robots": [
                        {
                            "status": "available",
                            "fields": {
                                "control_dim": 6,
                                "input_min": low.tolist(),
                                "input_max": high.tolist(),
                                "output_min": (
                                    output_mid + (low[:3] - input_mid) * gain
                                ).tolist(),
                                "output_max": (
                                    output_mid + (high[:3] - input_mid) * gain
                                ).tolist(),
                                "use_delta": True,
                                "control_delta": True,
                            },
                        }
                    ],
                }
                forecast = record["near_far_result"]
                gap = forecast["distance_samples"][0]
                pair_ids = record["near_far_geom_bindings"]
                objects = record["semantic_binding_validation"]["objects"]
                subject_body = objects[pair_ids["subject_object_id"]]["body_id"]
                obstacle_body = objects[pair_ids["obstacle_object_id"]]["body_id"]
                directions, _ = probe.derive_clearance_directions(
                    gap, subject_body, obstacle_body, up
                )
                state = RecoveryState()
                if step:
                    previous = np.flatnonzero(archive["step_indices"] == step - 1)
                    if len(previous) != 1:
                        raise ValueError("previous actual main action is unavailable")
                    state.last_actual_action = archive["executed_actions"][
                        previous[0]
                    ].tolist()
                specs, rejected = generate_recovery_candidates(
                    actions, directions, context, state
                )
                item.update({
                    "status": "generated_offline_only",
                    "controller_source": "archived validated Brake-and-Steer gain, affine midpoints, input bounds and lift direction; no live controller queried",
                    "binding_source": "historical decision binding metadata, not a new scene binding",
                    "generated_count": len(specs),
                    "constraint_rejected_count": len(rejected),
                    "generated_by_type": dict(
                        Counter(
                            entry["diagnostics"]["candidate_type"] for entry in specs
                        )
                    ),
                    "rejection_reason_counts": dict(
                        Counter(
                            entry["command_constraints"].get("reason")
                            for entry in rejected
                        )
                    ),
                    "exceeds_old_relative_budget_count": sum(
                        entry["diagnostics"].get("relative_A0_maximum_correction_m", 0)
                        > 0.032
                        or entry["diagnostics"].get("relative_A0_total_correction_m", 0)
                        > 0.08
                        for entry in specs
                        if entry["diagnostics"]["validation_policy"]
                        == "recovery_absolute_control"
                    ),
                    "stop_approach_diagnostics": next(
                        (
                            entry["diagnostics"]
                            for entry in specs
                            if entry["diagnostics"]["candidate_type"] == "stop_approach"
                        ),
                        None,
                    ),
                    "not_verified": [
                        "physical contacts",
                        "workspace/joint paths",
                        "object/EEF continuity",
                        "Snapshot restoration",
                        "real task progress",
                    ],
                })
            except (ValueError, KeyError, StopIteration, TypeError) as exc:
                item["reason"] = (
                    str(exc) or "required archived generation diagnostics are missing"
                )
    return results


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Offline Recovery candidate diagnostics; no model or simulator"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50, 55])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = diagnose_archive_generation(args.archive, args.steps)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
