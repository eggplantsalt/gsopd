"""Offline-recompile archived On-Policy branches with the current contact compiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read {label} JSON {path}: {exc}") from exc


def recompile_archive(
    archive_path: Path,
    teacher_path: Path,
    object_id: str,
    semantic_results_path: Path,
    geometry_inventory_path: Path,
    gripper_inventory_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    teacher = parse_teacher_output_v2(teacher_path.read_text(encoding="utf-8"))
    semantic_results = _read_json(semantic_results_path, "object semantic results")
    geometry_inventory = _read_json(geometry_inventory_path, "Bridge geometry inventory")
    gripper_inventory = _read_json(gripper_inventory_path, "Bridge gripper inventory")
    if not isinstance(semantic_results, dict):
        raise ValueError("Object semantic results must be a JSON object keyed by Teacher object ID")
    if object_id not in semantic_results:
        raise ValueError(f"No semantic binding was supplied for object_id {object_id!r}")

    with np.load(archive_path, allow_pickle=False) as archive:
        if "decision_safety_json" not in archive.files:
            raise ValueError("Archive is missing decision_safety_json")
        raw_decisions = archive["decision_safety_json"]
        decisions = []
        for index, raw in enumerate(raw_decisions):
            try:
                decision = json.loads(str(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Archive decision {index} contains invalid JSON") from exc
            summary = decision.get("a0_branch_summary")
            monitoring = summary.get("contact_monitoring") if isinstance(summary, dict) else None
            if not isinstance(monitoring, dict):
                decisions.append(
                    {
                        "decision_index": index,
                        "decision_step": decision.get("decision_step"),
                        "old_a0_verdict": decision.get("a0_verdict"),
                        "new_verdict": "unknown",
                        "reason": "archived_branch_contact_monitoring_is_missing",
                    }
                )
                continue
            executed_steps = summary.get("executed_steps")
            if isinstance(executed_steps, bool) or not isinstance(executed_steps, int) or executed_steps <= 0:
                decisions.append(
                    {
                        "decision_index": index,
                        "decision_step": decision.get("decision_step"),
                        "old_a0_verdict": decision.get("a0_verdict"),
                        "new_verdict": "unknown",
                        "reason": "archived_branch_executed_step_count_is_invalid",
                    }
                )
                continue
            candidate = {
                "candidate_id": f"archive-decision-{index}",
                "status": summary.get("status", "ok"),
                "executed_steps": executed_steps,
                "steps": [{"step_index": step} for step in range(executed_steps)],
                "contact_monitoring": monitoring,
            }
            branch_result = {
                "main_state_restored": decision.get("main_state_restored") is True,
                "candidates": [candidate],
            }
            compiled = compile_forbid_contact(
                teacher,
                object_id,
                semantic_results[object_id],
                branch_result,
                geometry_inventory,
                gripper_inventory,
                object_semantic_results=semantic_results,
            )
            decisions.append(
                {
                    "decision_index": index,
                    "decision_step": decision.get("decision_step"),
                    "old_a0_verdict": decision.get("a0_verdict"),
                    "new_verdict": compiled["status"],
                    "compiled": compiled,
                }
            )

    counts = {
        verdict: sum(item["new_verdict"] == verdict for item in decisions)
        for verdict in ("safe", "violation", "unknown")
    }
    result = {
        "archive": str(archive_path.resolve()),
        "teacher_json": str(teacher_path.resolve()),
        "object_id": object_id,
        "semantic_results": str(semantic_results_path.resolve()),
        "geometry_inventory": str(geometry_inventory_path.resolve()),
        "gripper_inventory": str(gripper_inventory_path.resolve()),
        "decision_count": len(decisions),
        "verdict_counts": counts,
        "decisions": decisions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--teacher-json", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--semantic-results-json", type=Path, required=True)
    parser.add_argument("--geometry-inventory", type=Path, required=True)
    parser.add_argument("--gripper-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for label, path in (
        ("archive", args.archive),
        ("Teacher JSON", args.teacher_json),
        ("semantic results", args.semantic_results_json),
        ("geometry inventory", args.geometry_inventory),
        ("gripper inventory", args.gripper_inventory),
    ):
        if not path.is_file():
            parser.error(f"{label} file does not exist: {path}")
    try:
        result = recompile_archive(
            args.archive,
            args.teacher_json,
            args.object_id,
            args.semantic_results_json,
            args.geometry_inventory,
            args.gripper_inventory,
            args.out,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Offline contact recompile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Recompiled {result['decision_count']} decisions: {result['verdict_counts']}")
    for decision in result["decisions"]:
        if decision["new_verdict"] == "violation":
            print(
                f"decision={decision['decision_index']} step={decision['decision_step']} "
                f"old={decision['old_a0_verdict']} new=violation"
            )
    print(f"Results saved: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
