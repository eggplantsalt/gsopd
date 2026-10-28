"""Read-only discovery of GS-OPD grasp, operation, and keypose priors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.gs_opd.repair.task_skill_experience import (
    _object_family,
    load_experience_library,
    matching_grasp_templates,
    matching_operation_skill,
)

ROOT = Path("/opt/data/private/gs_opd/experience")
LIBRARY = ROOT / "gs_opd_task_skill_experience_library_20261026_v2.json"
KEYPOSES = (
    ROOT / "gs_opd_l0t2_keypose_experience_20261026_v1.json",
    ROOT / "gs_opd_l1t2_keypose_experience_20261026_v1.json",
)


def query_experience(
    object_id,
    relation,
    region,
    task,
    *,
    library=LIBRARY,
    keyposes=KEYPOSES,
):
    if not all(
        isinstance(x, str) and x.strip()
        for x in (object_id, relation, region, task)
    ):
        raise ValueError("object_id_relation_region_task_required")

    source = load_experience_library(Path(library))

    if source.get("source_policy", {}).get("source_is_GS_OPD_owned") is not True:
        raise ValueError("experience_library_not_GS_OPD_owned")

    if source.get("source_episode", {}).get("native_task_success") is not True:
        raise ValueError("experience_library_source_task_not_successful")

    family = _object_family(object_id)
    grasps = matching_grasp_templates(source, object_id)
    operation = matching_operation_skill(source, object_id)

    if any(
        x.get("official_safety_demonstration_data_used") is True
        for x in grasps
    ):
        raise ValueError("official_grasp_demonstration_prohibited")

    if (
        operation
        and operation.get("official_safety_demonstration_data_used") is True
    ):
        raise ValueError("official_operation_demonstration_prohibited")

    candidates = []

    for path in keyposes:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.get("schema_version") != "gs_opd_keypose_experience.v1":
            raise ValueError(f"unsupported_keypose_schema:{path}")

        src = data.get("source", {})

        if src.get("official_safety_demonstrations_used") is True:
            raise ValueError(f"official_keypose_demonstration_prohibited:{path}")

        if src.get("native_task_success") is not True:
            raise ValueError(f"unverified_keypose_source:{path}")

        if (
            data.get("object_family") != family
            or str(data.get("target_relation", "")).lower() != relation.lower()
            or data.get("target_region_name") != region
        ):
            continue

        candidates.append({
            "id": data.get("experience_id"),
            "file": str(path),
            "source_task": src.get("task"),
            "source_safety_success": src.get("safety_success"),
            "same_task": task in str(src.get("task", "")),
            "keyposes": [
                x.get("name")
                for x in data.get("keypose_sequence", [])
            ],
        })

    candidates.sort(
        key=lambda x: (
            not x["same_task"],
            x["source_task"] or "",
        )
    )

    return {
        "object_id": object_id,
        "object_family": family,
        "relation": relation,
        "region": region,
        "task": task,
        "grasp_template_ids": [
            x.get("template_id")
            for x in grasps
        ],
        "operation_skill_id": (
            operation.get("skill_id")
            if operation else None
        ),
        "operation_phases": (
            operation.get("phase_sequence", [])
            if operation else []
        ),
        "keypose_candidates": candidates,
        "matched": bool(grasps or operation or candidates),
        "library_source_operational_safety": (
            source["source_episode"].get("operational_safety_pass")
        ),
        "library_source_strict_10mm": (
            source["source_episode"].get("strict_10mm")
        ),
        "live_scene_BDDL_verified": False,
        "source_hashes_reverified": False,
        "new_scene_physics_verified": False,
        "new_scene_safety_certified": False,
        "training_label_valid": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--task", required=True)

    args = parser.parse_args()

    result = query_experience(
        args.object_id,
        args.relation,
        args.region,
        args.task,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
