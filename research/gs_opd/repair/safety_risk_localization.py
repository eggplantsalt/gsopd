# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Localize observed substep risks without treating missing measurements as safe."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _geometry_contact(substep: dict[str, Any], geom_a: int, geom_b: int) -> dict[str, Any] | None:
    if not isinstance(substep, dict):
        return None
    for row in substep.get("contacts", []):
        if not isinstance(row, dict):
            continue
        ids = {row.get("geom1", {}).get("id"), row.get("geom2", {}).get("id")}
        if ids == {geom_a, geom_b}:
            return {
                "position_world_m": row.get("position"),
                "normal_geom1_to_geom2_world": row.get("normal_geom1_to_geom2"),
                "normal_geom_order": [row.get("geom1", {}).get("id"), row.get("geom2", {}).get("id")],
                "signed_contact_distance_m": row.get("signed_contact_distance_m"),
                "force_and_torque": row.get("force_and_torque"),
            }
    return None


def localize_risks(
    records: Iterable[dict[str, Any]], policy: dict[str, Any],
    *, snapshots: list[dict[str, Any]] | None = None,
    teacher_attested: bool = False,
) -> dict[str, Any]:
    """Return chronological observed windows and explicit coverage limitations.

    The policy must be independently derived from the current BDDL and any
    available teacher rules. The function never promotes diagnostic status to
    a safety verdict; it evaluates pair evidence against the supplied rules.
    """
    groups = list(policy.get("forbidden_objects", [])) + list(policy.get("allowed_target_objects", []))
    pair_rules = policy.get("forbidden_object_pairs", [])
    pad_rules = policy.get("allowed_contact_pairs", [])
    if not isinstance(pair_rules, list) or not isinstance(pad_rules, list) or not groups:
        raise ValueError("risk_localization_requires_explicit_bound_geometry_policy")
    pad_by_key = {}
    for row in pad_rules:
        a, b = sorted((int(row["robot_geom_id"]), int(row["scene_geom_id"])))
        threshold = float(row["minimum_clearance_m"])
        if not math.isfinite(threshold) or threshold < -0.002:
            raise ValueError("pad_penetration_limit_must_not_exceed_2mm")
        pad_by_key[f"{a}:{b}"] = (a, b, threshold)
    pair_sets = [
        (str(row["object_id_a"]), str(row["object_id_b"]),
         set(row["geom_ids_a"]), set(row["geom_ids_b"])) for row in pair_rules
    ]
    snapshots = sorted(snapshots or [], key=lambda row: row["timestep"])
    windows: list[dict[str, Any]] = []
    previous_global_substep = None
    previous_timestep = 0
    missing_substeps = 0
    missing_object_pair_distances = 0
    actions = 0
    for record in records:
        timestep = record.get("timestep")
        if type(timestep) is not int or timestep != previous_timestep + 1:
            raise ValueError("risk_records_are_not_a_continuous_t0_episode")
        previous_timestep = timestep
        actions += 1
        monitor = record.get("contact_monitoring", {})
        geometry = monitor.get("geometry_monitoring", {})
        rows = geometry.get("substeps", []) if isinstance(geometry, dict) else []
        contacts = monitor.get("substep_contacts", [])
        if (monitor.get("status") != "complete" or monitor.get("captured_physics_substeps") != 25
                or len(rows) != 25 or not isinstance(contacts, list) or len(contacts) != 25):
            raise ValueError(f"risk_substep_records_incomplete_at_timestep:{timestep}")
        for substep_index, (row, contact_row) in enumerate(zip(rows, contacts)):
            global_substep = (timestep - 1) * 25 + substep_index
            violations = []
            incomplete = (not isinstance(row, dict) or row.get("status") not in {"complete", "unsafe"}
                          or not isinstance(contact_row, dict)
                          or contact_row.get("status") not in {"contact", "no_contact"}
                          or contact_row.get("physics_substep_index") != substep_index)
            if not isinstance(row, dict):
                row = {}
            for field, rule in (("self_contacts", "robot_self_contact"),
                                ("strict_joint_limit_violations", "strict_joint_limit")):
                entries = row.get(field)
                if not isinstance(entries, list):
                    incomplete = True
                else:
                    for entry in entries:
                        violations.append({"rule": rule, "evidence": entry})
            unresolved = row.get("unresolved_contacts")
            if not isinstance(unresolved, list) or unresolved:
                incomplete = True
            clearance = row.get("clearances_by_object", {})
            for group in groups:
                object_id = group["object_id"]
                measurement = clearance.get(object_id, {}) if isinstance(clearance, dict) else {}
                distance = measurement.get("minimum_distance_m") if isinstance(measurement, dict) else None
                if (measurement.get("status") != "complete" or not isinstance(distance, (int, float))
                        or isinstance(distance, bool) or not math.isfinite(float(distance))):
                    incomplete = True
                    continue
                margin = max(0.010, float(group.get("minimum_clearance_m", 0.010)))
                if float(distance) < margin:
                    pair = measurement.get("pair") or {}
                    violations.append({
                        "rule": "robot_non_authorized_clearance", "object_id": object_id,
                        "distance_m": float(distance), "minimum_required_m": margin,
                        "geometry_pair": pair,
                        "contact": _geometry_contact(contact_row, pair.get("robot_geom_id"), pair.get("object_geom_id")),
                    })
            pad_values = row.get("allowed_contact_clearances_m", {})
            for key, (robot_id, object_id, minimum) in pad_by_key.items():
                distance = pad_values.get(key) if isinstance(pad_values, dict) else None
                if not isinstance(distance, (int, float)) or isinstance(distance, bool) or not math.isfinite(float(distance)):
                    incomplete = True
                    continue
                if float(distance) < minimum:
                    violations.append({
                        "rule": "authorized_pad_penetration", "geometry_ids": [robot_id, object_id],
                        "distance_m": float(distance), "minimum_required_m": minimum,
                        "contact": _geometry_contact(contact_row, robot_id, object_id),
                    })
            measured_pairs = row.get("forbidden_object_pair_clearances")
            if pair_sets and (not isinstance(measured_pairs, dict)
                              or measured_pairs.get("measurement_complete") is not True):
                incomplete = True
                missing_object_pair_distances += 1
            if isinstance(measured_pairs, dict):
                actual = {(p.get("object_id_a"), p.get("object_id_b")) for p in measured_pairs.get("pairs", [])}
                if actual != {(a, b) for a, b, _, _ in pair_sets}:
                    incomplete = True
                for pair in measured_pairs.get("pairs", []):
                    distance = pair.get("measurement", {}).get("minimum_distance_m")
                    required = pair.get("minimum_required_m")
                    if (not isinstance(distance, (int, float)) or isinstance(distance, bool)
                            or not math.isfinite(float(distance)) or not isinstance(required, (int, float))
                            or isinstance(required, bool) or not math.isfinite(float(required))
                            or float(required) < 0.010):
                        incomplete = True
                        continue
                    violation = float(distance) < float(required)
                    if pair.get("verdict") != ("violation" if violation else "pass"):
                        incomplete = True
                    if violation:
                        geom_pair = pair.get("geometry_pair") or {}
                        violations.append({
                            "rule": "forbidden_object_pair_clearance",
                            "objects": [pair.get("object_id_a"), pair.get("object_id_b")],
                            "distance_m": pair.get("measurement", {}).get("minimum_distance_m"),
                            "minimum_required_m": pair.get("minimum_required_m"),
                            "geometry_pair": geom_pair,
                            "contact": _geometry_contact(contact_row, geom_pair.get("geom_a_id"), geom_pair.get("geom_b_id")),
                        })
            for contact in contact_row.get("contacts", []) if isinstance(contact_row, dict) else []:
                g1, g2 = contact.get("geom1", {}).get("id"), contact.get("geom2", {}).get("id")
                for name_a, name_b, geoms_a, geoms_b in pair_sets:
                    if (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b):
                        violations.append({
                            "rule": "forbidden_object_pair_contact", "objects": [name_a, name_b],
                            "geometry_ids": [g1, g2], "distance_m": contact.get("signed_contact_distance_m"),
                            "minimum_required_m": 0.010, "contact": _geometry_contact(contact_row, g1, g2),
                        })
            if incomplete:
                missing_substeps += 1
            if violations:
                location = {"timestep": timestep, "physics_substep_index": substep_index,
                            "phase": record.get("phase"), "global_substep_index": global_substep}
                if not windows or previous_global_substep != global_substep - 1:
                    eligible = [s for s in snapshots if s.get("timestep", -1) < timestep]
                    windows.append({"start": location, "end": location, "violating_substeps": 0,
                                    "first_evidence": violations[:8], "rules": [], "objects": [],
                                    "nearest_preceding_snapshot": eligible[-1] if eligible else None})
                window = windows[-1]
                window["end"] = location
                window["violating_substeps"] += 1
                for violation in violations:
                    rule = violation["rule"]
                    if rule not in window["rules"]:
                        window["rules"].append(rule)
                    for name in ([violation["object_id"]] if violation.get("object_id") else violation.get("objects", [])):
                        if name and name not in window["objects"]:
                            window["objects"].append(name)
                previous_global_substep = global_substep
    return {
        "schema_version": "gs_opd_observed_risk_windows.v1",
        "actions": actions, "physics_substeps": actions * 25,
        "risk_windows": windows, "first_observed_risk": windows[0] if windows else None,
        "incomplete_substeps": missing_substeps,
        "missing_forbidden_object_pair_distance_substeps": missing_object_pair_distances,
        "teacher_attested": teacher_attested,
        # Scene-contact policy provenance has not yet been independently
        # matched for every segment. Never certify complete coverage.
        "earliest_strict_risk_proven": False,
        "strict_safety_pass": False if windows else None,
        "status": "incomplete",
        "coverage_limitations": ["scene_contact_policy_not_independently_verified_for_every_segment"],
        "source": policy.get("source"),
        "note": "A preceding snapshot is not a certified safe state unless all prior safety evidence and its restore identity are verified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    snapshots = json.loads(args.snapshots.read_text(encoding="utf-8")) if args.snapshots else []

    def records():
        for path in args.records:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)

    result = localize_risks(records(), policy, snapshots=snapshots, teacher_attested=False)
    result["record_archives"] = [str(path.resolve()) for path in args.records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "risk_windows": len(result["risk_windows"]),
                      "incomplete_substeps": result["incomplete_substeps"]}))


if __name__ == "__main__":
    main()
