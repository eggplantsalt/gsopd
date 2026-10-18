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
"""Read actual planning geometry and screen locally derived bypass paths."""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import numpy as np

from research.gs_opd.repair.trajectory_bundle import (
    BundleConfig,
    generate_trajectory_bundle,
    save_trajectory_bundle,
)
from research.gs_opd.smoke.near_far_gap_smoke import _select_forbid_pair
from research.gs_opd.smoke.replay_archive_contact_smoke import (
    _bridge_request,
    _validate_current_bindings,
)
from research.gs_opd.teacher.schema_v2 import (
    parse_teacher_output_v2,
    validate_teacher_output_v2,
)


def _selected_bindings(teacher, semantic_results, inventory, subject_id, instruction):
    validation = _validate_current_bindings(teacher, semantic_results, inventory)
    if validation.get("valid") is not True:
        raise ValueError(
            f"current_geometry_binding_validation_failed:{validation.get('errors')}"
        )
    rules = [
        rule
        for rule in teacher.get("object_contact_rules", [])
        if rule.get("decision") == "forbid" and rule.get("predicate") == "contact"
    ]
    subjects = {rule["subject_id"] for rule in rules}
    if subject_id is None:
        if len(subjects) != 1:
            raise ValueError(
                "An unambiguous carried subject or --subject-object-id is required"
            )
        subject_id = next(iter(subjects))
    obstacles = sorted({
        rule["target_id"] for rule in rules if rule["subject_id"] == subject_id
    })
    if not obstacles:
        raise ValueError("Selected subject has no forbidden-contact obstacle rules")
    for obstacle_id in obstacles:
        _select_forbid_pair(teacher, subject_id, obstacle_id)
    declarations = {
        semantic_results[key]["task_instruction"]
        for key in [subject_id, *obstacles]
        if isinstance(semantic_results[key].get("task_instruction"), str)
        and semantic_results[key]["task_instruction"]
    }
    if instruction is not None:
        if not instruction or (declarations and declarations != {instruction}):
            raise ValueError(
                "Expected instruction conflicts with semantic binding metadata"
            )
    elif len(declarations) == 1:
        instruction = next(iter(declarations))
    else:
        raise ValueError(
            "expected_instruction_unavailable_or_conflicting; provide --expected-instruction"
        )

    def binding(key):
        value = validation["objects"][key]
        return {
            "object_id": key,
            **{
                name: value[name]
                for name in ("body_id", "body_name", "geom_ids", "geom_names")
            },
        }

    return {
        "op": "planning_geometry",
        "carried_binding": binding(subject_id),
        "obstacle_bindings": [binding(key) for key in obstacles],
        "expected_instruction": instruction,
    }, validation


def _check_query(query, teacher, semantic_results, payload):
    if (
        query.get("status") != "complete"
        or query.get("query_complete") is not True
        or query.get("physical_state_unchanged") is not True
        or query.get("scratch_forward_used") is not True
    ):
        raise ValueError(f"planning_geometry_query_incomplete:{query.get('error')}")
    if (
        query.get("frame") != "world"
        or query.get("units") != "m"
        or query.get("instruction") != payload["expected_instruction"]
    ):
        raise ValueError("planning_geometry_frame_or_instruction_mismatch")
    state_id = query.get("state_id")
    if (
        not isinstance(state_id, str)
        or not state_id
        or not query.get("model_geometry_sha256")
    ):
        raise ValueError("planning_geometry_state_identity_missing")
    if any(
        query.get("state_consistency", {}).get(name) is not True
        for name in (
            "integration_state_equal",
            "model_geometry_equal",
            "live_kinematics_cache_equal",
        )
    ):
        raise ValueError("planning_geometry_nonmutation_not_proven")
    state = np.asarray(query.get("mujoco_integration_state"), dtype=np.float64)
    if state.ndim != 1 or not len(state) or not np.isfinite(state).all():
        raise ValueError("planning_geometry_integration_state_invalid")
    checked = _validate_current_bindings(
        teacher, semantic_results, query.get("geometry_inventory")
    )
    if checked.get("valid") is not True:
        raise ValueError("planning_geometry_exact_model_binding_validation_failed")
    objects = query.get("objects", {})
    for binding in [payload["carried_binding"], *payload["obstacle_bindings"]]:
        item = objects.get(binding["object_id"], {})
        if (
            item.get("body_id") != binding["body_id"]
            or item.get("body_name") != binding["body_name"]
            or item.get("state_id") != state_id
            or item.get("binding_verified") is not True
            or item.get("enclosure_verified") is not True
            or item.get("covers_all_associated_collision_geoms") is not True
            or item.get("semantic_binding_geom_ids") != binding["geom_ids"]
        ):
            raise ValueError(
                "planning_geometry_full_object_identity_or_enclosure_unconfirmed"
            )
        geom_ids, components = item.get("geom_ids"), item.get("components")
        center = np.asarray(item.get("center_world_m"), dtype=float)
        radius = item.get("radius_m")
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not np.isfinite(radius)
            or radius <= 0
            or not isinstance(geom_ids, list)
            or not geom_ids
            or len(set(geom_ids)) != len(geom_ids)
            or not isinstance(components, list)
            or len(components) != len(geom_ids)
            or {component.get("geom_id") for component in components} != set(geom_ids)
        ):
            raise ValueError("planning_geometry_sphere_or_component_coverage_invalid")
        inventory = {
            entry["geom_id"]: entry
            for entry in query["geometry_inventory"]["geometries"]
        }
        for component in components:
            geom_id = component["geom_id"]
            entry = inventory.get(geom_id, {})
            for name in ("geom_name", "body_id", "body_name"):
                if component.get(name) != entry.get(name):
                    raise ValueError("planning_geometry_component_identity_mismatch")
            position = np.asarray(component.get("center_world_m"), dtype=float)
            component_radius = component.get("radius_m")
            if (
                position.shape != (3,)
                or not np.isfinite(position).all()
                or isinstance(component_radius, bool)
                or not isinstance(component_radius, (int, float))
                or not np.isfinite(component_radius)
                or component_radius <= 0
                or np.linalg.norm(position - center) + component_radius > radius + 1e-12
            ):
                raise ValueError("planning_geometry_component_not_enclosed")
    return checked


def _compare_queries(before, after):
    left = np.asarray(before["mujoco_integration_state"], dtype=float)
    right = np.asarray(after["mujoco_integration_state"], dtype=float)
    same = (
        left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
    )
    checks = {
        "state_id_equal": before["state_id"] == after["state_id"],
        "model_geometry_equal": before["model_geometry_sha256"]
        == after["model_geometry_sha256"],
        "integration_state_exactly_equal": bool(same and np.array_equal(left, right)),
        "instruction_equal": before["instruction"] == after["instruction"],
        "object_bounds_equal": before["objects"] == after["objects"],
        "goal_diagnostics_equal": before.get("goal") == after.get("goal"),
    }
    return {
        "matched": all(checks.values()),
        "checks": checks,
        "maximum_integration_state_error": float(np.abs(left - right).max())
        if same
        else None,
    }


class _LocalGoalUnavailable(ValueError):
    def __init__(self, reason, diagnostics=None):
        super().__init__(reason)
        self.diagnostics = diagnostics or []


def _build_local_bypass_scene(
    query, payload, *, clearance_m=0.01, exit_clearance_m=0.02
):
    """Build a local endpoint from same-state spheres and the active BDDL On target."""
    if (
        isinstance(clearance_m, bool)
        or not isinstance(clearance_m, (int, float))
        or not math.isfinite(float(clearance_m))
        or clearance_m < 0
    ):
        raise ValueError("local_bypass_goal_unavailable:clearance_m_invalid")
    if (
        isinstance(exit_clearance_m, bool)
        or not isinstance(exit_clearance_m, (int, float))
        or not math.isfinite(float(exit_clearance_m))
        or exit_clearance_m < 0
    ):
        raise ValueError("local_bypass_goal_unavailable:exit_clearance_m_invalid")
    clearance_m = float(clearance_m)
    exit_clearance_m = float(exit_clearance_m)
    state_id = query["state_id"]
    objects = query["objects"]
    carried = objects[payload["carried_binding"]["object_id"]]
    carried_task_key = carried.get("registered_task_object")
    if not isinstance(carried_task_key, str) or not carried_task_key:
        raise ValueError("local_bypass_goal_unavailable:carried_bddl_object_missing")

    goal_diagnostics = query.get("goal", {})
    native_goals = goal_diagnostics.get("native_goal_predicates")
    if not isinstance(native_goals, list):
        raise ValueError("local_bypass_goal_unavailable:active_bddl_goals_missing")
    on_goals = [
        item
        for item in native_goals
        if isinstance(item, (list, tuple))
        and len(item) == 3
        and str(item[0]).lower() == "on"
        and item[1] == carried_task_key
    ]
    if len(on_goals) != 1:
        raise ValueError(
            "local_bypass_goal_unavailable:active_bddl_on_target_missing_or_ambiguous"
        )
    predicate = list(on_goals[0])
    target_key = predicate[2]
    target_diagnostics = [
        item
        for item in goal_diagnostics.get("targets", [])
        if isinstance(item, dict)
        and item.get("predicate") == predicate
        and (
            item.get("registered_task_object") == target_key
            or item.get("native_target_site_key") == target_key
        )
    ]
    if len(target_diagnostics) != 1:
        raise ValueError(
            "local_bypass_goal_unavailable:compiled_bddl_target_body_missing_or_ambiguous"
        )
    target = target_diagnostics[0]
    target_registered_key = target.get("registered_task_object")
    target_body_id, target_body_name = target.get("body_id"), target.get("body_name")
    target_position = np.asarray(
        target.get("body_origin_world_m_diagnostic_only"), dtype=np.float64
    )
    if (
        isinstance(target_body_id, bool)
        or not isinstance(target_body_id, int)
        or target_body_id < 0
        or not isinstance(target_body_name, str)
        or not target_body_name
        or target_position.shape != (3,)
        or not np.isfinite(target_position).all()
        or target.get("body_origin_is_carried_center_goal") is not False
    ):
        raise ValueError("local_bypass_goal_unavailable:target_body_diagnostic_invalid")
    inventory_geometries = query.get("geometry_inventory", {}).get("geometries", [])
    if not any(
        entry.get("body_id") == target_body_id
        and entry.get("body_name") == target_body_name
        and entry.get("category") == "scene"
        for entry in inventory_geometries
    ):
        raise ValueError(
            "local_bypass_goal_unavailable:target_body_not_in_current_geometry_inventory"
        )
    bound_body_ids = {
        carried.get("body_id"),
        *(
            objects[item["object_id"]].get("body_id")
            for item in payload["obstacle_bindings"]
        ),
    }
    if target_body_id in bound_body_ids:
        raise ValueError(
            "local_bypass_goal_unavailable:target_body_conflicts_with_bound_object"
        )

    start = np.asarray(carried["center_world_m"], dtype=np.float64)
    carried_radius = float(carried["radius_m"])
    target_vector = target_position - start
    target_distance = float(np.linalg.norm(target_vector))
    if not math.isfinite(target_distance) or target_distance <= 1e-9:
        raise ValueError("local_bypass_goal_unavailable:placement_direction_degenerate")
    forward = target_vector / target_distance

    obstacle_candidates = []
    diagnostics = []
    for binding in payload["obstacle_bindings"]:
        obstacle = objects[binding["object_id"]]
        center = np.asarray(obstacle["center_world_m"], dtype=np.float64)
        radius = float(obstacle["radius_m"])
        offset = center - start
        along = float(np.dot(offset, forward))
        lateral = float(np.linalg.norm(offset - along * forward))
        minimum_separation = carried_radius + radius + clearance_m
        required_separation = minimum_separation + exit_clearance_m
        endpoint_along = along + required_separation
        reasons = []
        if not (0 < along < target_distance):
            reasons.append("obstacle_not_between_carried_state_and_bddl_target")
        if lateral > required_separation:
            reasons.append("obstacle_outside_forward_carried_sphere_corridor")
        if endpoint_along >= target_distance:
            reasons.append("clearance_endpoint_would_reach_or_pass_placement_target")
        diagnostics.append(
            {
                "obstacle_object_id": binding["object_id"],
                "obstacle_center_world_m": center.tolist(),
                "projection_from_start_m": along,
                "lateral_distance_from_task_direction_m": lateral,
                "required_center_separation_m": required_separation,
                "safety_clearance_m": clearance_m,
                "exit_clearance_m": exit_clearance_m,
                "rejection_reasons": reasons,
            }
        )
        if not reasons:
            obstacle_candidates.append((along, obstacle, required_separation))
    if not obstacle_candidates:
        raise _LocalGoalUnavailable(
            "local_bypass_goal_unavailable:no_forbidden_obstacle_in_forward_corridor",
            diagnostics,
        )
    # The first physically relevant obstacle along the BDDL task direction is
    # the local bypass reference; ties retain the validated binding order.
    obstacle_projection, obstacle, required_separation = min(
        obstacle_candidates, key=lambda item: item[0]
    )
    scale = max(
        1.0, required_separation, float(np.abs(obstacle["center_world_m"]).max())
    )
    numeric_pad = 1e-8 * scale
    endpoint_separation = required_separation + numeric_pad
    obstacle_center = np.asarray(obstacle["center_world_m"], dtype=np.float64)
    local_goal_position = obstacle_center + forward * endpoint_separation
    conservative_gap = (
        float(np.linalg.norm(local_goal_position - obstacle_center))
        - carried_radius
        - float(obstacle["radius_m"])
    )
    if (
        not np.isfinite(local_goal_position).all()
        or conservative_gap < clearance_m + exit_clearance_m
        or obstacle_projection + endpoint_separation >= target_distance
    ):
        raise ValueError(
            "local_bypass_goal_unavailable:computed_endpoint_geometry_is_invalid"
        )

    goal = {
        "goal_type": "local_bypass_goal",
        "position_world_m": local_goal_position.tolist(),
        "position_semantics": "carried_sphere_center",
        "verified": True,
        "verification_scope": "same_state_geometry_derived_waypoint_only_not_final_placement_or_physical_safety",
        "source": "same-state conservative carried/obstacle spheres; direction from active BDDL On target Body origin only",
        "state_id": state_id,
        "construction": {
            "obstacle_object_id": obstacle["object_id"],
            "obstacle_center_world_m": obstacle_center.tolist(),
            "forward_world": forward.tolist(),
            "placement_target_direction_reference": {
                "body_id": target_body_id,
                "body_name": target_body_name,
                "body_origin_world_m": target_position.tolist(),
                "used_as_final_goal": False,
                "registered_task_object": target_registered_key,
                "native_target_key": target_key,
                "native_predicate": predicate,
            },
            "carried_registered_task_object": carried_task_key,
            "direction_source": "active_bddl_on_target_body_origin_direction_only",
            "obstacle_projection_from_start_m": obstacle_projection,
            "obstacle_lateral_distance_m": next(
                item["lateral_distance_from_task_direction_m"]
                for item in diagnostics
                if item["obstacle_object_id"] == obstacle["object_id"]
            ),
            "obstacle_selection_rule": "nearest_eligible_forbidden_obstacle_along_task_direction",
            "required_sphere_center_separation_m": required_separation,
            "safety_clearance_m": clearance_m,
            "endpoint_sphere_center_separation_m": endpoint_separation,
            "endpoint_conservative_sphere_gap_m": conservative_gap,
            "clearance_margin_m": clearance_m,
            "exit_clearance_m": exit_clearance_m,
            "numeric_padding_m": numeric_pad,
            "all_obstacle_forward_checks": diagnostics,
        },
    }
    scene = {
        "frame": "world",
        "units": "m",
        "state_id": state_id,
        "world_up": query["world_up"],
        "carried": carried,
        "obstacles": [
            objects[item["object_id"]] for item in payload["obstacle_bindings"]
        ],
        "goal": goal,
    }
    return scene


def run_scene_smoke(
    connection: Any,
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    out_dir: str | Path,
    *,
    subject_id: str | None = None,
    expected_instruction: str | None = None,
    exit_clearance_m: float = 0.02,
) -> dict[str, Any]:
    """Query twice without reset/step; screen local bypass proposals if derivable."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reserved = (
        "scene_smoke_report.json",
        "planning_geometry.json",
        "planning_geometry_after.json",
        "scene.json",
        "trajectory_bundle.json",
        "trajectory_bundle.svg",
    )
    if any((directory / name).exists() for name in reserved):
        raise FileExistsError(
            "Use a fresh output directory to avoid mixing geometry from different states"
        )
    report = {
        "status": "unavailable",
        "geometry_query_succeeded": False,
        "physical_safety_verified": False,
        "trajectory_executability_verified": False,
        "final_placement_goal_verified": False,
        "goal_status": "unavailable",
        "requested_candidate_count": 18,
        "local_bypass_exit_clearance_m": exit_clearance_m,
        "candidate_count": 0,
        "screen_pass_count": None,
        "rejection_reason_counts": {},
        "current_visual_reidentification_performed": False,
        "automatic_environment_actions_executed": 0,
    }
    payload = first = bundle = None
    first_valid = False
    try:
        teacher = validate_teacher_output_v2(teacher)
        inventory = _bridge_request(connection, {"op": "geometry_inventory"})
        payload, validation = _selected_bindings(
            teacher, semantic_results, inventory, subject_id, expected_instruction
        )
        report.update({"binding_validation": validation, "query_payload": payload})
        first = _bridge_request(connection, payload)
        report["planning_geometry"] = first
        report["same_query_binding_validation"] = _check_query(
            first, teacher, semantic_results, payload
        )
        first_valid = True
        bundle_config = BundleConfig(candidate_count=18)
        try:
            scene = _build_local_bypass_scene(
                first,
                payload,
                clearance_m=bundle_config.clearance_m,
                exit_clearance_m=exit_clearance_m,
            )
        except ValueError as exc:
            report.update(
                {
                    "status": "goal_unavailable",
                    "goal_status": "goal_unavailable",
                    "goal_diagnostics": first.get("goal", {}),
                    "goal_construction_diagnostics": getattr(
                        exc, "diagnostics", []
                    ),
                    "reason": str(exc),
                }
            )
        else:
            report["scene"] = scene
            bundle = generate_trajectory_bundle(scene, bundle_config)
            report.update(
                {
                    "status": "complete",
                    "goal_status": "local_bypass_goal_constructed",
                    "final_placement_goal_verified": False,
                    "physical_safety_verified": False,
                    "trajectory_executability_verified": False,
                    "trajectory_bundle_generated": True,
                }
            )
    except Exception as exc:
        report.update({
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        if first is not None and payload is not None:
            try:
                after = _bridge_request(connection, payload)
                report["planning_geometry_after"] = after
                _check_query(after, teacher, semantic_results, payload)
                comparison = _compare_queries(first, after)
                report["state_consistency"] = comparison
                if comparison["matched"] is not True:
                    raise ValueError(
                        "Planning geometry/state changed during the read-only Smoke"
                    )
                report["geometry_query_succeeded"] = (
                    first.get("query_complete") is True
                    and first.get("physical_state_unchanged") is True
                )
            except Exception as exc:
                report.update({
                    "status": "state_consistency_failed"
                    if first_valid
                    else "unavailable",
                    "geometry_query_succeeded": False,
                    "state_consistency_error": f"{type(exc).__name__}: {exc}",
                })
    if report["status"] == "complete" and bundle is not None:
        report["bundle_outputs"] = save_trajectory_bundle(bundle, directory)
        report.update({
            name: bundle[name]
            for name in (
                "candidate_count",
                "screen_pass_count",
                "screen_rejected_count",
                "rejection_reason_counts",
            )
        })
        (directory / "scene.json").write_text(
            json.dumps(report["scene"], ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    for key, filename in (
        ("planning_geometry", "planning_geometry.json"),
        ("planning_geometry_after", "planning_geometry_after.json"),
    ):
        if key in report:
            (directory / filename).write_text(
                json.dumps(report[key], ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
    (directory / "scene_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Connect to the existing Bridge; never reset or step its environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=38567)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--semantic-results-json", type=Path, required=True)
    parser.add_argument("--subject-object-id")
    parser.add_argument("--expected-instruction")
    parser.add_argument(
        "--exit-clearance-m",
        type=float,
        default=0.02,
        help="additional local bypass endpoint space beyond the sphere safety clearance (default: 0.02 m)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    teacher = parse_teacher_output_v2(args.teacher_json.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_results_json.read_text(encoding="utf-8"))
    connection = Client((args.host, args.port), authkey=b"gs-opd-local")
    try:
        report = run_scene_smoke(
            connection,
            teacher,
            semantic,
            args.out_dir,
            subject_id=args.subject_object_id,
            expected_instruction=args.expected_instruction,
            exit_clearance_m=args.exit_clearance_m,
        )
    finally:
        connection.close()
    sys.stdout.write(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "geometry_query_succeeded",
                    "candidate_count",
                    "screen_pass_count",
                    "screen_rejected_count",
                    "reason",
                    "rejection_reason_counts",
                    "error",
                    "state_consistency_error",
                    "trajectory_executability_verified",
                    "goal_status",
                    "final_placement_goal_verified",
                    "local_bypass_exit_clearance_m",
                )
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    sys.stdout.write(f"Report: {args.out_dir / 'scene_smoke_report.json'}\n")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
