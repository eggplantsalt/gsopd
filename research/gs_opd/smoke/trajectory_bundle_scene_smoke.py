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
"""Read actual planning geometry; generate curves only with a proven center goal."""

from __future__ import annotations

import argparse
import json
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


def _verified_scene(query, payload):
    goal = query.get("goal", {})
    if (
        goal.get("status") != "complete"
        or goal.get("verified") is not True
        or goal.get("safe_center_region_verified") is not True
        or goal.get("state_id") != query["state_id"]
        or goal.get("position_semantics") != "carried_sphere_center"
        or not isinstance(goal.get("source"), str)
        or not goal["source"]
    ):
        return None
    target = goal.get("placement_target", {})
    if not any(
        entry.get("body_id") == target.get("body_id")
        and entry.get("body_name") == target.get("body_name")
        and entry.get("category") == "scene"
        for entry in query["geometry_inventory"]["geometries"]
    ):
        raise ValueError("verified_goal_target_body_not_in_current_geometry_inventory")
    return {
        "frame": "world",
        "units": "m",
        "state_id": query["state_id"],
        "world_up": query["world_up"],
        "carried": query["objects"][payload["carried_binding"]["object_id"]],
        "obstacles": [
            query["objects"][item["object_id"]] for item in payload["obstacle_bindings"]
        ],
        "goal": goal,
    }


def run_scene_smoke(
    connection: Any,
    teacher: dict[str, Any],
    semantic_results: dict[str, Any],
    out_dir: str | Path,
    *,
    subject_id: str | None = None,
    expected_instruction: str | None = None,
) -> dict[str, Any]:
    """Query twice without reset/step; save bounds even if the native goal is unknown."""
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
        "requested_candidate_count": 18,
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
        scene = _verified_scene(first, payload)
        if scene is None:
            report.update({
                "status": "goal_unavailable",
                "reason": first.get("goal", {}).get(
                    "reason", "No proven safe carried-center goal region"
                ),
            })
        else:
            report["scene"] = scene
            bundle = generate_trajectory_bundle(scene, BundleConfig(candidate_count=18))
            report["status"] = "complete"
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
                    "reason",
                    "error",
                    "state_consistency_error",
                    "trajectory_executability_verified",
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
