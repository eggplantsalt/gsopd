import argparse
import json
import os
from pathlib import Path
from multiprocessing.connection import Client

os.environ["EMBODIED_PATH"] = "/workspace/zy_workspace/RLinf/examples/embodiment"

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.models import get_model
from rlinf.models.embodiment.openpi.checkpoint import (
    resolve_full_weights,
    resolve_model_safetensors,
)
from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft"),
        help="OpenPI model checkpoint directory or weights file.",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        help="Optional standalone Action Expert LoRA adapter paired with --checkpoint.",
    )
    parser.add_argument(
        "--record-physical-state",
        action="store_true",
        help="Record read-only MuJoCo object states and post-step contacts in the NPZ trace.",
    )
    parser.add_argument(
        "--record-substep-contacts",
        action="store_true",
        help="Passively capture every native MuJoCo physics-substep contact for Safety Compiler auditing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional Python/NumPy/PyTorch RNG seed for a reproducible Student rollout.",
    )
    parser.add_argument(
        "--result-out",
        type=Path,
        default=Path(
            "/opt/data/private/gs_opd/rollouts/pi05_safety_l0t0_result.json"
        ),
        help="JSON path for the final rollout summary.",
    )
    parser.add_argument(
        "--video-out",
        type=Path,
        help="Optional MP4 path for the agentview rollout video.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=240,
        help="Maximum number of environment actions to execute (default: 240).",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        help="Optional NPZ path for on-policy rollout transitions.",
    )
    parser.add_argument(
        "--grasp-only-smoke",
        action="store_true",
        help="Run an isolated raw-Base grasp Smoke and stop after three distinct stable-grasp states.",
    )
    parser.add_argument(
        "--grasp-smoke-object-id",
        default="object_1",
        help="Reviewed semantic object ID used by --grasp-only-smoke (default: object_1).",
    )
    parser.add_argument(
        "--contact-safety-probe",
        action="store_true",
        help="Probe the first predicted Action Chunk with Bridge branching, then stop before real actions.",
    )
    parser.add_argument(
        "--verifier-guided-repair",
        action="store_true",
        help="Search Snapshot-verified local XYZ repairs for the first predicted Action Chunk, then stop.",
    )
    parser.add_argument(
        "--on-policy-safety-collection",
        action="store_true",
        help="Verify Student chunks, execute safe local recovery when needed, and collect actual transitions.",
    )
    parser.add_argument(
        "--trajectory-bundle-physical-validation",
        action="store_true",
        help="Run one Snapshot-only trajectory tracking acceptance after a live stable grasp is observed.",
    )
    parser.add_argument("--teacher-json", type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--semantic-result", type=Path)
    parser.add_argument(
        "--semantic-results-json",
        type=Path,
        help="Optional mapping from Teacher object IDs to current SAM/IoU semantic bindings for On-Policy collection.",
    )
    parser.add_argument("--probe-out-dir", type=Path)
    parser.add_argument("--repair-out-dir", type=Path)
    parser.add_argument("--on-policy-safety-out", type=Path)
    parser.add_argument("--task-object-id")
    parser.add_argument("--task-object-semantic-result", type=Path)
    parser.add_argument("--trajectory-subject-object-id")
    parser.add_argument("--trajectory-validation-out", type=Path)
    parser.add_argument(
        "--mink-osc-left",
        action="store_true",
        help="Plan one current-state left route with Mink and verify native OSC in a Snapshot branch after stable grasp.",
    )
    parser.add_argument(
        "--direct-bspline-osc-left",
        action="store_true",
        help="After stable grasp, execute the default left B-Spline route, or the route selected by --bspline-candidate-id, with native OSC_POSE.",
    )
    parser.add_argument(
        "--bspline-candidate-id",
        help="Optional candidate_id from the current geometrically screened B-Spline bundle (requires --direct-bspline-osc-left).",
    )
    parser.add_argument(
        "--bspline-left-001-toward-004-interp-10pct",
        action="store_true",
        help="Compatibility alias for --bspline-left-001-toward-004-fraction 0.10.",
    )
    parser.add_argument(
        "--bspline-left-001-toward-004-fraction",
        type=float,
        help="Interpolate this fraction of the current-state trajectory-001 Left control points toward trajectory-004, then rerun the existing geometric screen (0 < fraction <= 1).",
    )
    parser.add_argument("--mink-python", type=Path)
    parser.add_argument("--mink-model-binary", type=Path)
    parser.add_argument("--mink-model-metadata", type=Path)
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be zero or greater")
    if args.record_substep_contacts and not args.record_physical_state:
        parser.error("--record-substep-contacts requires --record-physical-state")
    if args.video_out is not None and args.video_out.suffix != ".mp4":
        parser.error("--video-out must use the .mp4 extension")
    if args.trace_out is not None and args.trace_out.suffix != ".npz":
        parser.error("--trace-out must use the .npz extension")
    probe_inputs = (
        args.teacher_json,
        args.object_id,
        args.semantic_result,
    )
    if args.contact_safety_probe and args.verifier_guided_repair:
        parser.error("--contact-safety-probe and --verifier-guided-repair are mutually exclusive")
    probe_enabled = args.contact_safety_probe or args.verifier_guided_repair
    collection_enabled = args.on_policy_safety_collection
    trajectory_validation_enabled = args.trajectory_bundle_physical_validation
    mink_osc_enabled = args.mink_osc_left
    direct_osc_enabled = args.direct_bspline_osc_left
    grasp_smoke_enabled = args.grasp_only_smoke
    interpolation_fraction = args.bspline_left_001_toward_004_fraction
    if args.bspline_left_001_toward_004_interp_10pct:
        if interpolation_fraction is not None:
            parser.error(
                "use either --bspline-left-001-toward-004-interp-10pct or "
                "--bspline-left-001-toward-004-fraction, not both"
            )
        interpolation_fraction = 0.10
    if interpolation_fraction is not None and (
        not np.isfinite(interpolation_fraction)
        or not 0.0 < interpolation_fraction <= 1.0
    ):
        parser.error("--bspline-left-001-toward-004-fraction must be finite and in (0, 1]")
    args.bspline_left_001_toward_004_fraction = interpolation_fraction
    if interpolation_fraction is not None and not direct_osc_enabled:
        parser.error(
            "B-Spline control-point interpolation requires --direct-bspline-osc-left"
        )
    if interpolation_fraction is not None and args.bspline_candidate_id is not None:
        parser.error(
            "B-Spline control-point interpolation cannot be combined with --bspline-candidate-id"
        )
    if args.bspline_candidate_id is not None and not direct_osc_enabled:
        parser.error("--bspline-candidate-id requires --direct-bspline-osc-left")
    if args.bspline_candidate_id is not None and not args.bspline_candidate_id.strip():
        parser.error("--bspline-candidate-id must not be empty")
    if sum((probe_enabled, collection_enabled, trajectory_validation_enabled)) > 1:
        parser.error("probe, collection, and trajectory physical validation modes are mutually exclusive")
    if grasp_smoke_enabled and (
        probe_enabled
        or collection_enabled
        or trajectory_validation_enabled
        or mink_osc_enabled
        or direct_osc_enabled
    ):
        parser.error("--grasp-only-smoke cannot be combined with probe, collection, trajectory, Mink, or OSC modes")
    if mink_osc_enabled and direct_osc_enabled:
        parser.error("--mink-osc-left and --direct-bspline-osc-left are mutually exclusive")
    if mink_osc_enabled and not trajectory_validation_enabled:
        parser.error("--mink-osc-left requires --trajectory-bundle-physical-validation")
    if direct_osc_enabled and not trajectory_validation_enabled:
        parser.error("--direct-bspline-osc-left requires --trajectory-bundle-physical-validation")
    if grasp_smoke_enabled:
        original_sft_base = Path(
            "/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft"
        ).resolve()
        if args.checkpoint.expanduser().resolve() != original_sft_base:
            parser.error("--grasp-only-smoke requires the original pi05_rlinf_libero_sft Base")
        if args.lora_checkpoint is not None:
            parser.error("--grasp-only-smoke must not load a LoRA Adapter")
        if args.max_steps == 0 or args.max_steps > 200:
            parser.error("--grasp-only-smoke requires --max-steps in [1, 200]")
        if args.teacher_json is None or args.semantic_results_json is None:
            parser.error("--grasp-only-smoke requires --teacher-json and --semantic-results-json")
        if args.semantic_result is not None or args.object_id is not None:
            parser.error("--grasp-only-smoke uses the complete mapping; omit --semantic-result/--object-id")
        if args.video_out is None or args.trace_out is None:
            parser.error("--grasp-only-smoke requires --video-out and --trace-out")
        if not args.record_physical_state:
            parser.error("--grasp-only-smoke requires --record-physical-state")
        if any(
            value is not None
            for value in (
                args.probe_out_dir,
                args.repair_out_dir,
                args.on_policy_safety_out,
                args.task_object_id,
                args.task_object_semantic_result,
                args.trajectory_subject_object_id,
                args.trajectory_validation_out,
                args.mink_python,
                args.mink_model_binary,
                args.mink_model_metadata,
            )
        ):
            parser.error("--grasp-only-smoke does not accept unrelated probe or trajectory options")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if not args.semantic_results_json.is_file():
            parser.error(f"Semantic results mapping does not exist: {args.semantic_results_json}")
    elif trajectory_validation_enabled:
        if args.teacher_json is None or args.semantic_results_json is None:
            parser.error(
                "--trajectory-bundle-physical-validation requires --teacher-json and --semantic-results-json"
            )
        if args.semantic_result is not None or args.object_id is not None:
            parser.error("trajectory physical validation uses the full v2 semantic mapping; omit --semantic-result/--object-id")
        if args.trajectory_validation_out is None:
            parser.error("--trajectory-bundle-physical-validation requires --trajectory-validation-out")
        if args.trajectory_validation_out.suffix != ".json":
            parser.error("--trajectory-validation-out must use the .json extension")
        if args.max_steps == 0:
            parser.error("trajectory physical validation requires --max-steps greater than zero")
        if mink_osc_enabled:
            if any(
                value is None
                for value in (
                    args.mink_python,
                    args.mink_model_binary,
                    args.mink_model_metadata,
                )
            ):
                parser.error("--mink-osc-left requires --mink-python, --mink-model-binary, and --mink-model-metadata")
            if not args.mink_python.is_file() or not args.mink_model_binary.is_file() or not args.mink_model_metadata.is_file():
                parser.error("Mink Python and current-model export files must exist")
        if (args.trace_out is not None and not direct_osc_enabled) or args.on_policy_safety_out is not None:
            parser.error("trajectory physical validation does not write rollout trace/training data unless direct OSC execution is enabled")
        if direct_osc_enabled and (args.trace_out is None or args.video_out is None):
            parser.error("--direct-bspline-osc-left requires --trace-out and --video-out for real action/observation recording")
        if args.probe_out_dir is not None or args.repair_out_dir is not None:
            parser.error("probe and repair output directories cannot be used for trajectory validation")
        if args.task_object_id is not None or args.task_object_semantic_result is not None:
            parser.error("trajectory physical validation gets task bindings from its complete semantic mapping")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if not args.semantic_results_json.is_file():
            parser.error(f"Semantic results mapping does not exist: {args.semantic_results_json}")
        if args.trajectory_validation_out.exists():
            parser.error(f"Choose a fresh trajectory validation report path: {args.trajectory_validation_out}")
    elif collection_enabled:
        if args.trajectory_validation_out is not None or args.trajectory_subject_object_id is not None:
            parser.error("trajectory validation options require --trajectory-bundle-physical-validation")
        if args.teacher_json is None or args.object_id is None:
            parser.error(
                "--on-policy-safety-collection requires --teacher-json and --object-id"
            )
        if args.semantic_result is not None and args.semantic_results_json is not None:
            parser.error("choose either --semantic-result or --semantic-results-json")
        if args.semantic_result is None and args.semantic_results_json is None:
            parser.error(
                "--on-policy-safety-collection requires --semantic-result or --semantic-results-json"
            )
        if args.on_policy_safety_out is None:
            parser.error("--on-policy-safety-collection requires --on-policy-safety-out")
        if args.on_policy_safety_out.suffix != ".npz":
            parser.error("--on-policy-safety-out must use the .npz extension")
        if args.probe_out_dir is not None or args.repair_out_dir is not None:
            parser.error("probe and repair output directories cannot be used for episode collection")
        if args.trace_out is not None:
            parser.error("--trace-out cannot be combined with --on-policy-safety-collection")
        if args.max_steps == 0:
            parser.error("--on-policy-safety-collection requires --max-steps greater than zero")
        if args.semantic_results_json is None and (
            (args.task_object_id is None) != (args.task_object_semantic_result is None)
        ):
            parser.error("--task-object-id and --task-object-semantic-result must be supplied together")
        if args.semantic_results_json is not None and args.task_object_semantic_result is not None:
            parser.error("--semantic-results-json already supplies the task-object binding; omit --task-object-semantic-result")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if args.semantic_result is not None and not args.semantic_result.is_file():
            parser.error(f"Semantic result file does not exist: {args.semantic_result}")
        if args.semantic_results_json is not None and not args.semantic_results_json.is_file():
            parser.error(f"Semantic results mapping does not exist: {args.semantic_results_json}")
        if args.task_object_semantic_result is not None and not args.task_object_semantic_result.is_file():
            parser.error(f"Task object semantic result does not exist: {args.task_object_semantic_result}")
    elif probe_enabled:
        if mink_osc_enabled or any(
            value is not None
            for value in (args.mink_python, args.mink_model_binary, args.mink_model_metadata)
        ):
            parser.error("Mink OSC options require --trajectory-bundle-physical-validation")
        if args.trajectory_validation_out is not None or args.trajectory_subject_object_id is not None:
            parser.error("trajectory validation options require --trajectory-bundle-physical-validation")
        if args.semantic_results_json is not None:
            parser.error("--semantic-results-json is currently supported only by --on-policy-safety-collection")
        if any(value is None for value in probe_inputs):
            parser.error(
                "a probe mode requires --teacher-json, --object-id, "
                "and --semantic-result"
            )
        if args.verifier_guided_repair and args.repair_out_dir is None:
            parser.error("--verifier-guided-repair requires --repair-out-dir")
        if args.on_policy_safety_out is not None:
            parser.error("--on-policy-safety-out requires --on-policy-safety-collection")
        if args.contact_safety_probe and args.probe_out_dir is None:
            parser.error("--contact-safety-probe requires --probe-out-dir")
        if args.verifier_guided_repair and args.probe_out_dir is not None:
            parser.error("--probe-out-dir is only used by --contact-safety-probe")
        if args.contact_safety_probe and args.repair_out_dir is not None:
            parser.error("--repair-out-dir requires --verifier-guided-repair")
        if args.max_steps == 0:
            parser.error("a probe mode requires --max-steps greater than zero")
        if args.trace_out is not None:
            parser.error("--trace-out cannot be combined with a branch probe mode")
        if not args.teacher_json.is_file():
            parser.error(f"Safety IR v2 file does not exist: {args.teacher_json}")
        if not args.semantic_result.is_file():
            parser.error(f"Semantic result file does not exist: {args.semantic_result}")
        if (args.task_object_id is None) != (args.task_object_semantic_result is None):
            parser.error("--task-object-id and --task-object-semantic-result must be supplied together")
        if args.task_object_id is not None and not args.verifier_guided_repair:
            parser.error("task-object binding options require --verifier-guided-repair")
        if args.task_object_semantic_result is not None and not args.task_object_semantic_result.is_file():
            parser.error(f"Task object semantic result does not exist: {args.task_object_semantic_result}")
    elif any(value is not None for value in probe_inputs) or any(
        value is not None for value in (
            args.probe_out_dir,
            args.repair_out_dir,
            args.on_policy_safety_out,
            args.semantic_results_json,
            args.task_object_id,
            args.task_object_semantic_result,
            args.trajectory_validation_out,
            args.trajectory_subject_object_id,
            args.mink_python,
            args.mink_model_binary,
            args.mink_model_metadata,
            True if args.mink_osc_left else None,
            True if args.direct_bspline_osc_left else None,
        )
    ):
        parser.error(
            "Teacher, binding, and output arguments require a probe mode"
        )
    if not trajectory_validation_enabled and (
        mink_osc_enabled
        or any(
            value is not None
            for value in (args.mink_python, args.mink_model_binary, args.mink_model_metadata)
        )
    ):
        parser.error("Mink OSC options require --trajectory-bundle-physical-validation")
    return args


def run_first_chunk_contact_probe(
    conn,
    chunk,
    instruction,
    teacher_json_path,
    object_id,
    semantic_result_path,
    out_dir,
):
    """Branch the complete first predicted chunk and compile its contact evidence."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.smoke.contact_safety_compiler_smoke import (
        compile_forbid_contact,
    )
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
    if not any(item["id"] == object_id for item in teacher["objects"]):
        raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
    semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            return {"error": "Safety Bridge returned a non-object response"}
        return response

    actions = chunk.tolist()
    candidates = [{"candidate_id": "pi05-first-action-chunk", "actions": actions}]
    geometry_inventory = {"status": "unavailable"}
    gripper_inventory = {"status": "unavailable"}
    branch_result = {"error": "branch_rollout was not started", "main_state_restored": False}
    snapshot_id = None
    try:
        geometry_inventory = bridge_request({"op": "geometry_inventory"})
        gripper_inventory = bridge_request({"op": "gripper_geometry_inventory"})
        snapshot = bridge_request({"op": "snapshot"})
        snapshot_id = snapshot.get("snapshot_id")
        if isinstance(snapshot_id, str):
            branch_result = bridge_request(
                {
                    "op": "branch_rollout",
                    "snapshot_id": snapshot_id,
                    "candidates": candidates,
                }
            )
        else:
            branch_result = {
                "error": snapshot.get("error", "Snapshot operation returned no ID"),
                "main_state_restored": False,
                "candidates": [],
            }
    except (OSError, EOFError) as exc:
        branch_result = {
            "error": f"Safety Bridge request failed: {type(exc).__name__}: {exc}",
            "main_state_restored": False,
            "candidates": [],
        }

    compiled_result = compile_forbid_contact(
        teacher,
        object_id,
        semantic_result,
        branch_result,
        geometry_inventory,
        gripper_inventory,
    )
    report = {
        "mode": "first_action_chunk_contact_safety_probe",
        "instruction": instruction,
        "object_id": object_id,
        "predicted_action_chunk": actions,
        "snapshot_id": snapshot_id,
        "main_state_restored": branch_result.get("main_state_restored") is True,
        "verdict": compiled_result["status"],
        "compiled_result": compiled_result,
        "bridge_geometry_inventory": geometry_inventory,
        "bridge_gripper_geometry_inventory": gripper_inventory,
        "branch_rollout_result": branch_result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "first_chunk_contact_safety_probe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "FIRST CHUNK CONTACT SAFETY PROBE:",
        report["verdict"],
        "MAIN STATE RESTORED:",
        report["main_state_restored"],
        flush=True,
    )
    print("PROBE RESULT:", report_path, flush=True)
    return report


def _resolve_task_object_binding(teacher, object_id, semantic_result, geometry_inventory):
    """Require explicit target_object semantics and a model-verified Body binding."""
    teacher_object = next((item for item in teacher["objects"] if item["id"] == object_id), None)
    if teacher_object is None or teacher_object.get("role") != "target_object":
        raise ValueError("task progress object must have Safety IR role target_object")
    if not isinstance(semantic_result, dict):
        raise ValueError("task object semantic result is not a JSON object")
    decision = semantic_result.get("semantic_decision")
    semantic_object = semantic_result.get("teacher_object")
    association = semantic_result.get("physical_association")
    if (
        not isinstance(decision, dict)
        or decision.get("object_id") != object_id
        or decision.get("status") != "resolved"
        or not isinstance(semantic_object, dict)
        or semantic_object.get("object_id") != object_id
        or not isinstance(association, dict)
        or association.get("status") != "matched_by_iou"
    ):
        raise ValueError("task object semantic result is unresolved or lacks a physical association")
    body_id = association.get("body_id")
    body_name = association.get("body_name")
    if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 0:
        raise ValueError("task object semantic binding has an invalid Body ID")
    if not isinstance(geometry_inventory, dict) or geometry_inventory.get("status") != "complete":
        raise ValueError("active Bridge geometry inventory is incomplete")
    body_entries = {
        item.get("body_id"): item
        for item in geometry_inventory.get("geometries", [])
        if isinstance(item, dict) and item.get("body_id") == body_id
    }
    if not body_entries or any(item.get("category") != "scene" for item in body_entries.values()):
        raise ValueError("task object Body ID is absent from the active scene model")
    active_names = {item.get("body_name") for item in body_entries.values()}
    if not isinstance(body_name, str) or active_names != {body_name}:
        raise ValueError("task object Body name disagrees with the active model")
    return {
        "status": "matched_by_iou",
        "object_id": object_id,
        "role": "target_object",
        "body_id": body_id,
        "body_name": body_name,
        "physical_object_id": association.get("physical_object_id"),
    }


def run_first_chunk_verifier_guided_repair(
    conn,
    chunk,
    instruction,
    teacher_json_path,
    object_id,
    semantic_result_path,
    out_dir,
    task_object_id=None,
    task_object_semantic_result_path=None,
):
    """Branch A0 and adaptively search verified repairs; never step the main env."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.repair.verifier_guided_repair import run_verifier_guided_repair
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            return {"error": "Safety Bridge returned a non-object response"}
        return response

    actions = np.asarray(chunk, dtype=np.float32)
    report = {
        "mode": "first_chunk_verifier_guided_action_repair_probe",
        "instruction": instruction,
        "object_id": object_id,
        "task_object_id": task_object_id,
        "original_action_chunk_A0": actions.tolist(),
        "snapshot_id": None,
        "main_state_restored": None,
        "status": "unknown",
        "repaired_action_chunk_A_plus": None,
        "bridge_geometry_inventory": {"status": "unavailable"},
        "bridge_gripper_geometry_inventory": {"status": "unavailable"},
        "bridge_repair_context": {"status": "unavailable"},
        "baseline_branch_result": None,
        "baseline_compiled_result": None,
        "repair_search": None,
    }
    try:
        teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
        if not any(item["id"] == object_id for item in teacher["objects"]):
            raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
        semantic_result = json.loads(semantic_result_path.read_text(encoding="utf-8"))
        task_semantic_result = (
            json.loads(task_object_semantic_result_path.read_text(encoding="utf-8"))
            if task_object_semantic_result_path is not None
            else None
        )
        report["bridge_geometry_inventory"] = bridge_request({"op": "geometry_inventory"})
        report["bridge_gripper_geometry_inventory"] = bridge_request({"op": "gripper_geometry_inventory"})
        report["bridge_repair_context"] = bridge_request({"op": "repair_context"})
        if report["bridge_repair_context"].get("status") != "available":
            raise RuntimeError("Safety Bridge did not provide active controller repair context")
        snapshot = bridge_request({"op": "snapshot"})
        report["snapshot_id"] = snapshot.get("snapshot_id")
        if not isinstance(report["snapshot_id"], str):
            raise RuntimeError(snapshot.get("error", "Safety Bridge Snapshot returned no ID"))
        if task_object_id is not None:
            report["task_object_binding"] = _resolve_task_object_binding(
                teacher,
                task_object_id,
                task_semantic_result,
                report["bridge_geometry_inventory"],
            )
        else:
            report["task_object_binding"] = None

        baseline_candidate_id = "repair-A0"
        baseline_branch = bridge_request({
            "op": "branch_rollout",
            "snapshot_id": report["snapshot_id"],
            "candidates": [{"candidate_id": baseline_candidate_id, "actions": actions.tolist()}],
        })
        report["baseline_branch_result"] = baseline_branch
        report["main_state_restored"] = baseline_branch.get("main_state_restored") is True
        baseline_compiled = compile_forbid_contact(
            teacher,
            object_id,
            semantic_result,
            baseline_branch,
            report["bridge_geometry_inventory"],
            report["bridge_gripper_geometry_inventory"],
        )
        report["baseline_compiled_result"] = baseline_compiled
        if not report["main_state_restored"]:
            report["status"] = "aborted_main_state_restore_failed"
            report["failure_reason"] = "baseline_branch_did_not_restore_the_preprobe_main_state"
        else:
            def evaluate_candidates(specifications):
                # One branch request per adaptive round; the Bridge restores the
                # same supplied Snapshot before each candidate and restores its
                # call-entry state in a finally block.
                report["main_state_restored"] = False
                branch = bridge_request({
                    "op": "branch_rollout",
                    "snapshot_id": report["snapshot_id"],
                    "candidates": [
                        {"candidate_id": item["candidate_id"], "actions": item["actions"]}
                        for item in specifications
                    ],
                })
                report["main_state_restored"] = branch.get("main_state_restored") is True
                compiled = compile_forbid_contact(
                    teacher,
                    object_id,
                    semantic_result,
                    branch,
                    report["bridge_geometry_inventory"],
                    report["bridge_gripper_geometry_inventory"],
                )
                branch_by_id = {
                    item.get("candidate_id"): item
                    for item in branch.get("candidates", [])
                    if isinstance(item, dict)
                }
                compiled_by_id = {
                    item.get("candidate_id"): item
                    for item in compiled.get("candidates", [])
                    if isinstance(item, dict)
                }
                results = []
                for item in specifications:
                    candidate_id = item["candidate_id"]
                    candidate_branch = branch_by_id.get(candidate_id)
                    candidate_compiled = compiled_by_id.get(candidate_id)
                    results.append({
                        "candidate_id": candidate_id,
                        "branch_result": {
                            **branch,
                            "candidates": [candidate_branch] if candidate_branch is not None else [],
                        },
                        "compiled_result": {
                            **compiled,
                            "candidates": [candidate_compiled] if candidate_compiled is not None else [],
                        },
                    })
                return results

            repair = run_verifier_guided_repair(
                actions,
                baseline_branch,
                baseline_compiled,
                report["bridge_repair_context"],
                evaluate_candidates,
                task_object_binding=report["task_object_binding"],
            )
            report["repair_search"] = repair
            report["status"] = repair.get("status", "unknown")
            report["repaired_action_chunk_A_plus"] = repair.get("repaired_action_chunk_A_plus")
            report["failure_reason"] = repair.get("failure_reason")
            report["main_state_restored"] = report["main_state_restored"] and repair.get("status") != "aborted_main_state_restore_failed"
    except Exception as exc:
        report["status"] = "aborted_unknown"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "verifier_guided_repair.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "VERIFIER-GUIDED REPAIR:", report["status"],
        "MAIN STATE RESTORED:", report["main_state_restored"],
        "HAS A+:", report["repaired_action_chunk_A_plus"] is not None,
        flush=True,
    )
    print("REPAIR RESULT:", report_path, flush=True)
    return report


def _compact_repair_report(repair_report):
    """Keep search decisions and actions while dropping duplicated branch payloads."""
    compact = dict(repair_report)

    def compact_candidate(candidate):
        if not isinstance(candidate, dict):
            return candidate
        return {
            key: value
            for key, value in candidate.items()
            if key not in ("branch_result", "compiled_result")
        }

    if isinstance(compact.get("search_attempts"), list):
        compact["search_attempts"] = [
            compact_candidate(item) for item in compact["search_attempts"]
        ]
    if "selected_candidate" in compact:
        compact["selected_candidate"] = compact_candidate(compact["selected_candidate"])
    return compact


def prepare_on_policy_safety_collection(
    conn,
    teacher_json_path,
    object_id,
    semantic_result_path=None,
    task_object_id=None,
    task_object_semantic_result_path=None,
    semantic_results_path=None,
):
    """Load fixed-task Teacher bindings and verify every referenced body/geom."""
    import json
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact
    from research.gs_opd.smoke.replay_archive_contact_smoke import (
        _validate_current_bindings,
    )
    from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

    def bridge_request(payload):
        conn.send(payload)
        response = conn.recv()
        if not isinstance(response, dict):
            raise RuntimeError("Safety Bridge returned a non-object response")
        return response

    teacher = parse_teacher_output_v2(teacher_json_path.read_text(encoding="utf-8"))
    if not any(item["id"] == object_id for item in teacher["objects"]):
        raise ValueError(f"object_id {object_id!r} is absent from Safety IR v2")
    if semantic_results_path is not None:
        loaded_results = json.loads(semantic_results_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_results, dict) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in loaded_results.items()
        ):
            raise ValueError("semantic results JSON must map Teacher object IDs to result objects")
        semantic_results = loaded_results
        if object_id not in semantic_results:
            raise ValueError(f"semantic results mapping has no binding for {object_id!r}")
    else:
        if semantic_result_path is None:
            raise ValueError("a semantic result file or semantic results mapping is required")
        semantic_results = {
            object_id: json.loads(semantic_result_path.read_text(encoding="utf-8"))
        }

    semantic_result = semantic_results[object_id]
    geometry_inventory = bridge_request({"op": "geometry_inventory"})
    gripper_inventory = bridge_request({"op": "gripper_geometry_inventory"})
    repair_context = bridge_request({"op": "repair_context"})

    object_contact_rules = teacher.get("object_contact_rules", [])
    binding_validation = None
    if object_contact_rules:
        binding_validation = _validate_current_bindings(
            teacher,
            semantic_results,
            geometry_inventory,
        )
        if binding_validation.get("valid") is not True:
            errors = binding_validation.get("errors") or [
                "object_semantic_bindings_are_unverified"
            ]
            raise ValueError(
                "Object-to-Object semantic Body/Geom bindings do not match the active "
                f"Geometry Inventory: {errors}"
            )

    # A branch error is expected in this read-only preflight. All other compiler
    # prerequisites must resolve before collecting from the fixed task.
    preflight = compile_forbid_contact(
        teacher,
        object_id,
        semantic_result,
        {"main_state_restored": True, "candidates": []},
        geometry_inventory,
        gripper_inventory,
        object_semantic_results=semantic_results,
    )
    preflight_errors = [
        reason
        for reason in preflight.get("unknown_reasons", [])
        if reason != "branch_rollout_has_no_candidate_results"
    ]
    selected_contact_rules = [
        rule
        for rule in teacher.get("rules", [])
        if isinstance(rule, dict)
        and rule.get("target_id") == object_id
        and rule.get("predicate") == "contact"
    ]
    if not selected_contact_rules and not object_contact_rules:
        preflight_errors.append("teacher_has_no_supported_contact_rules_for_collection")
    if selected_contact_rules:
        # The current robot-part compiler accepts exactly one gripper rule for
        # this target. An explicit allow is valid without the forbid-only
        # compiled_rule field; all other shapes remain fail-closed.
        if len(selected_contact_rules) != 1:
            preflight_errors.append("multiple_or_conflicting_gripper_contact_rules")
        else:
            selected_rule = selected_contact_rules[0]
            supported_gripper_rule = (
                selected_rule.get("subject_parts") == ["gripper"]
                and selected_rule.get("phase") == "any"
                and selected_rule.get("decision") in ("allow", "forbid")
            )
            if not supported_gripper_rule:
                preflight_errors.append("unsupported_or_unknown_gripper_contact_rule")
            elif (
                selected_rule["decision"] == "forbid"
                and preflight.get("compiled_rule") is None
            ):
                preflight_errors.append("gripper_contact_rule_did_not_compile")
    if (
        preflight.get("target_binding") is None
        or not preflight.get("gripper_geom_ids")
        or preflight_errors
    ):
        raise ValueError(
            "Safety IR rule or semantic physical binding is not valid for the active "
            f"LIBERO-Safety task: {preflight_errors or preflight.get('unknown_reasons')}"
        )

    task_object_binding = None
    if semantic_results_path is not None and task_object_id is None:
        inferred_targets = [
            item["id"] for item in teacher["objects"]
            if item.get("role") == "target_object"
        ]
        if object_contact_rules:
            if len(inferred_targets) != 1:
                raise ValueError(
                    "Object-to-Object collection requires exactly one target_object role "
                    "or an explicit --task-object-id"
                )
            task_object_id = inferred_targets[0]
        elif len(inferred_targets) == 1 and inferred_targets[0] in semantic_results:
            task_object_id = inferred_targets[0]

    if task_object_id is not None:
        if semantic_results_path is not None:
            task_semantic_result = semantic_results.get(task_object_id)
            if not isinstance(task_semantic_result, dict):
                raise ValueError(
                    "semantic results mapping has no task-progress binding for "
                    f"{task_object_id!r}"
                )
        else:
            if task_object_semantic_result_path is None:
                raise ValueError("task-object ID requires a semantic result binding")
            task_semantic_result = json.loads(
                task_object_semantic_result_path.read_text(encoding="utf-8")
            )
            semantic_results[task_object_id] = task_semantic_result
        task_object_binding = _resolve_task_object_binding(
            teacher,
            task_object_id,
            task_semantic_result,
            geometry_inventory,
        )
    return {
        "scope": {
            "suite": "obstacle_avoidance",
            "level": 0,
            "task_id": 0,
            "bridge_source": "research/gs_opd/smoke/safety_bridge.py",
            "binding_scope": "validated against the active fixed-task MuJoCo geometry inventory",
        },
        "teacher": teacher,
        "object_id": object_id,
        "semantic_result": semantic_result,
        "semantic_results": semantic_results,
        "semantic_binding_validation": binding_validation,
        "object_contact_rules": object_contact_rules,
        "task_object_id": task_object_id,
        "safety_target_binding": preflight["target_binding"],
        "geometry_inventory": geometry_inventory,
        "gripper_inventory": gripper_inventory,
        "repair_context": repair_context,
        "task_object_binding": task_object_binding,
        "preflight_result": preflight,
    }


def _near_far_result_is_complete(
    result, snapshot, actions, target_geom_ids, obstacle_geom_ids, decision_step
):
    """Fail closed unless the forecast covers this Snapshot/A0 and restored it."""
    if not isinstance(result, dict) or not isinstance(snapshot, dict):
        return False
    if result.get("status") != "complete" or result.get("risk_verdict") not in ("warning", "no_warning"):
        return False
    if result.get("risk_warning") is not (result.get("risk_verdict") == "warning"):
        return False
    if result.get("snapshot_id") != snapshot.get("snapshot_id"):
        return False
    try:
        from research.gs_opd.repair.near_far import action_chunk_fingerprint

        if result.get("action_chunk_sha256") != action_chunk_fingerprint(actions):
            return False
    except Exception:
        return False
    if (
        result.get("simulation_complete") is not True
        or result.get("requested_action_steps") != 5
        or result.get("executed_action_steps") != 5
    ):
        return False
    samples = result.get("distance_samples")
    if not isinstance(samples, list) or len(samples) != 6:
        return False
    for index, item in enumerate(samples):
        distance = item.get("minimum_distance_m") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("distance_available") is not True
            or item.get("step_index") != decision_step + index
            or isinstance(distance, bool)
            or not isinstance(distance, (int, float, np.integer, np.floating))
            or not np.isfinite(distance)
        ):
            return False
    restore = result.get("snapshot_restore")
    if not isinstance(restore, dict) or any(
        not isinstance(restore.get(key), dict) or restore[key].get("matched") is not True
        for key in ("before", "after")
    ):
        return False
    if not isinstance(target_geom_ids, list) or not isinstance(obstacle_geom_ids, list):
        return False
    try:
        return (
            set(result.get("target_geom_ids", [])) == set(target_geom_ids)
            and set(result.get("obstacle_geom_ids", [])) == set(obstacle_geom_ids)
            and not (set(target_geom_ids) & set(obstacle_geom_ids))
        )
    except TypeError:
        return False


def _validated_early_repair_candidates(result, expected_shape):
    """Recheck Early Repair's executable contract before any candidate can run."""
    from research.gs_opd.smoke.early_repair_s70_smoke import MIN_GAP_IMPROVEMENT_M

    def invalid(reason):
        return {"valid": False, "reason": reason, "candidates": [], "rejected_candidates": []}

    def finite_number(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float, np.integer, np.floating))
            and bool(np.isfinite(value))
        )

    if not isinstance(result, dict) or result.get("executable") is not True:
        return invalid("early_repair_did_not_report_executable")
    if not (
        result.get("verification_complete") is True
        and result.get("candidate_results_complete") is True
        and result.get("candidate_validation_complete") is True
        and result.get("candidate_branch_main_state_restored") is True
    ):
        return invalid("early_repair_candidate_verification_is_incomplete")
    restore = result.get("snapshot_restore")
    if not isinstance(restore, dict) or any(
        not isinstance(restore.get(key), dict) or restore[key].get("matched") is not True
        for key in ("before", "after")
    ):
        return invalid("early_repair_final_snapshot_restore_not_confirmed")
    if result.get("errors"):
        return invalid("early_repair_report_contains_errors")
    qualified = result.get("qualified_candidates")
    candidate_results = result.get("candidate_results")
    if not isinstance(qualified, list) or not qualified:
        return invalid("early_repair_has_no_qualified_candidates")
    if not isinstance(candidate_results, list) or not candidate_results:
        return invalid("early_repair_candidate_results_missing")
    by_id = {}
    for candidate in candidate_results:
        if not isinstance(candidate, dict):
            return invalid("early_repair_candidate_result_malformed")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in by_id:
            return invalid("early_repair_candidate_ids_missing_or_duplicate")
        by_id[candidate_id] = candidate
        if candidate.get("candidate_validation_complete") is not True:
            return invalid("early_repair_contains_incompletely_validated_candidate_results")

    validated = []
    rejected_candidates = []
    baseline_gaps = []
    seen = set()
    for item in qualified:
        if not isinstance(item, dict):
            return invalid("early_repair_qualified_candidate_malformed")
        candidate_id = item.get("candidate_id")
        verification = item.get("verification")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            return invalid("early_repair_qualified_candidate_ids_missing_or_duplicate")
        seen.add(candidate_id)
        candidate_result = by_id.get(candidate_id)
        if not isinstance(verification, dict) or candidate_result is None:
            return invalid("early_repair_qualified_candidate_has_no_matching_result")
        if not (
            verification.get("accepted_for_probe") is True
            and verification.get("candidate_validation_complete") is True
            and verification.get("full_five_step_chunk") is True
            and verification.get("contact_monitoring_complete") is True
            and verification.get("physical_safety_verdict") == "safe"
            and isinstance(verification.get("action_constraints"), dict)
            and verification["action_constraints"].get("status") == "pass"
            and isinstance(verification.get("task_progress"), dict)
            and verification["task_progress"].get("status") == "pass"
            and isinstance(verification.get("endpoint_replay"), dict)
            and verification["endpoint_replay"].get("status") == "complete"
            and verification["endpoint_replay"].get("main_state_restored") is True
        ):
            return invalid("early_repair_candidate_lacks_required_safety_progress_or_restore_checks")
        try:
            actions = np.asarray(item.get("actions"), dtype=np.float32)
            recorded = np.asarray(verification.get("actions"), dtype=np.float32)
            result_actions = np.asarray(candidate_result.get("actions"), dtype=np.float32)
        except (TypeError, ValueError):
            return invalid("early_repair_candidate_actions_are_not_numeric")
        if any(
            value.shape != expected_shape or not np.isfinite(value).all()
            for value in (actions, recorded, result_actions)
        ):
            return invalid("early_repair_candidate_action_shape_or_values_are_invalid")
        if not (
            np.array_equal(actions, recorded)
            and np.array_equal(actions, result_actions)
            and candidate_result.get("accepted_for_probe") is True
        ):
            return invalid("early_repair_candidate_actions_or_acceptance_do_not_match")
        endpoint_gap = verification.get("endpoint_gap_m")
        baseline_gap = verification.get("baseline_a0_endpoint_gap_m")
        reported_improvement = verification.get("endpoint_gap_change_vs_a0_m")
        result_gap = candidate_result.get("endpoint_gap_m")
        result_baseline_gap = candidate_result.get("baseline_a0_endpoint_gap_m")
        result_improvement = candidate_result.get("endpoint_gap_change_vs_a0_m")
        if not all(finite_number(value) for value in (
            endpoint_gap,
            baseline_gap,
            reported_improvement,
            result_gap,
            result_baseline_gap,
            result_improvement,
        )):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "endpoint_gap_or_a0_baseline_gap_is_missing_or_non_finite",
            })
            continue
        if not all(
            np.isclose(float(verified_value), float(result_value), rtol=1e-9, atol=1e-12)
            for verified_value, result_value in (
                (endpoint_gap, result_gap),
                (baseline_gap, result_baseline_gap),
                (reported_improvement, result_improvement),
            )
        ):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "qualified_candidate_gap_metadata_does_not_match_candidate_result",
            })
            continue
        gap_improvement = float(endpoint_gap) - float(baseline_gap)
        if (
            not np.isclose(float(reported_improvement), gap_improvement, rtol=1e-9, atol=1e-12)
            or gap_improvement <= MIN_GAP_IMPROVEMENT_M
        ):
            rejected_candidates.append({
                "candidate_id": candidate_id,
                "reason": "endpoint_gap_improvement_does_not_match_or_pass_existing_gate",
            })
            continue
        baseline_gaps.append(float(baseline_gap))
        validated.append({
            "candidate_id": candidate_id,
            "actions": actions,
            "endpoint_gap_m": float(endpoint_gap),
            "baseline_a0_endpoint_gap_m": float(baseline_gap),
            "gap_improvement_m": gap_improvement,
        })
    if not validated:
        return {
            "valid": False,
            "reason": "early_repair_has_no_candidates_with_finite_verified_gap_improvement",
            "candidates": [],
            "rejected_candidates": rejected_candidates,
        }
    baseline_gap = baseline_gaps[0]
    if any(not np.isclose(value, baseline_gap, rtol=1e-9, atol=1e-12) for value in baseline_gaps[1:]):
        return invalid("early_repair_candidates_disagree_on_a0_endpoint_gap")
    return {
        "valid": True,
        "reason": None,
        "candidates": validated,
        "rejected_candidates": rejected_candidates,
        "baseline_a0_endpoint_gap_m": baseline_gap,
    }


def _run_bias_collection_decision(
    conn,
    chunk,
    instruction,
    chunk_index,
    decision_step,
    collection_context,
    recovery_state=None,
):
    """Verify A0 from a fresh Snapshot and optionally search for a safe A+."""
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from research.gs_opd.repair.verifier_guided_repair import run_verifier_guided_repair
    from research.gs_opd.smoke.contact_safety_compiler_smoke import compile_forbid_contact

    actions = np.asarray(chunk, dtype=np.float32)
    object_id = collection_context["object_id"]
    teacher = collection_context["teacher"]
    semantic_result = collection_context["semantic_result"]
    semantic_results = collection_context.get("semantic_results") or {object_id: semantic_result}
    geometry_inventory = collection_context["geometry_inventory"]
    gripper_inventory = collection_context["gripper_inventory"]
    report = {
        "chunk_index": int(chunk_index),
        "decision_step": int(decision_step),
        "instruction": str(instruction),
        "teacher_context": teacher,
        "object_id": object_id,
        "rules": [rule for rule in teacher["rules"] if rule["target_id"] == object_id],
        "object_contact_rules": collection_context.get("object_contact_rules", []),
        "safety_target_binding": collection_context["safety_target_binding"],
        "semantic_binding_result": semantic_result,
        "semantic_binding_results": semantic_results,
        "semantic_binding_validation": collection_context.get("semantic_binding_validation"),
        "original_action_chunk_A0": actions.tolist(),
        "snapshot_id": None,
        "a0_verdict": "unknown",
        "a0_compiler_verdict": "unknown",
        "a0_compiled_result": None,
        "a0_branch_summary": None,
        "repair_attempted": False,
        "repair_status": None,
        "repair_failure_reason": None,
        "repair_search": None,
        "near_far_status": None,
        "near_far_risk_verdict": None,
        "near_far_result": None,
        "early_repair_status": None,
        "early_repair_search": None,
        "early_repair_selected_candidate_id": None,
        "selection_policy": "max_endpoint_gap_then_candidate_id_ascending",
        "selected_candidate_id": None,
        "selected_endpoint_gap_m": None,
        "baseline_a0_endpoint_gap_m": None,
        "selected_gap_improvement_m": None,
        "qualified_candidates_min_endpoint_gap_m": None,
        "qualified_candidates_max_endpoint_gap_m": None,
        "qualified_candidate_endpoint_gap_count": 0,
        "early_repair_candidate_gap_exclusions": [],
        "repaired_action_chunk_A_plus": None,
        "training_action_chunk": None,
        "training_label_valid": False,
        "label_source": None,
        "main_state_restored": None,
        "must_abort_episode": False,
        "failure_reasons": [],
    }

    def add_failure(reason):
        if reason is not None and str(reason) not in report["failure_reasons"]:
            report["failure_reasons"].append(str(reason))

    if actions.ndim != 2 or actions.shape[1:] != (7,) or not len(actions):
        report.update({
            "status": "invalid_A0_action_shape",
            "main_state_restored": True,
            "must_abort_episode": True,
        })
        add_failure("student_A0_must_have_shape_(T,7)")
        return report
    if not np.isfinite(actions).all():
        report.update({
            "status": "invalid_A0_action_values",
            "main_state_restored": True,
            "must_abort_episode": True,
        })
        add_failure("student_A0_contains_non_finite_values")
        return report

    def bridge_request(payload):
        try:
            conn.send(payload)
            response = conn.recv()
        except (OSError, EOFError) as exc:
            raise ConnectionError(
                f"Safety Bridge connection failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise RuntimeError("Safety Bridge returned a non-object response")
        return response

    a0_candidate_id = f"on-policy-A0-{int(chunk_index):06d}"
    try:
        snapshot = bridge_request({"op": "snapshot"})
        report["snapshot_id"] = snapshot.get("snapshot_id")
        if not isinstance(report["snapshot_id"], str):
            # Record the unknown decision; execution selection rejects it
            # before any unverified Student action can reach the environment.
            report["main_state_restored"] = True
            report["status"] = "unknown_snapshot_unavailable"
            add_failure(snapshot.get("error", "snapshot_operation_returned_no_id"))
            add_failure("original_action_chunk_not_branch_verified")
            return report

        baseline_branch = bridge_request({
            "op": "branch_rollout",
            "snapshot_id": report["snapshot_id"],
            "candidates": [{"candidate_id": a0_candidate_id, "actions": actions.tolist()}],
        })
        report["main_state_restored"] = baseline_branch.get("main_state_restored") is True
        baseline_candidate = next(
            (
                item for item in baseline_branch.get("candidates", [])
                if isinstance(item, dict) and item.get("candidate_id") == a0_candidate_id
            ),
            None,
        )
        if baseline_candidate is not None:
            branch_steps = baseline_candidate.get("steps")
            branch_step_count = len(branch_steps) if isinstance(branch_steps, list) else 0
            requested_steps = baseline_candidate.get("requested_steps")
            executed_steps = baseline_candidate.get("executed_steps")
            report["a0_branch_summary"] = {
                "status": baseline_candidate.get("status"),
                "requested_steps": requested_steps,
                "executed_steps": executed_steps,
                "step_record_count": branch_step_count,
                "full_action_chunk_executed": (
                    isinstance(requested_steps, int)
                    and not isinstance(requested_steps, bool)
                    and isinstance(executed_steps, int)
                    and not isinstance(executed_steps, bool)
                    and requested_steps == len(actions)
                    and executed_steps == len(actions)
                    and branch_step_count == len(actions)
                ),
                "step_rewards": [step.get("reward") for step in branch_steps or []],
                "step_dones": [step.get("done") for step in branch_steps or []],
                "contact_monitoring": baseline_candidate.get("contact_monitoring"),
            }
        compiled = compile_forbid_contact(
            teacher,
            object_id,
            semantic_result,
            baseline_branch,
            geometry_inventory,
            gripper_inventory,
            object_semantic_results=semantic_results,
        )
        report["a0_compiled_result"] = compiled
        report["a0_compiler_verdict"] = compiled.get("status", "unknown")
        report["a0_verdict"] = report["a0_compiler_verdict"]

        if report["main_state_restored"] is not True:
            report["status"] = "aborted_main_state_restore_failed"
            report["must_abort_episode"] = True
            add_failure("baseline_branch_did_not_restore_the_decision_state")
            return report

        a0_full_chunk = (
            isinstance(report["a0_branch_summary"], dict)
            and report["a0_branch_summary"].get("full_action_chunk_executed") is True
        )
        if report["a0_verdict"] == "safe" and not a0_full_chunk:
            report["a0_verdict"] = "unknown"
            report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
            add_failure("A0_safe_verdict_did_not_cover_the_requested_full_chunk")
            return report

        if recovery_state is not None and teacher.get("object_contact_rules"):
            from research.gs_opd.repair.local_recovery import (
                run_local_escape_and_recovery,
            )

            recovery = run_local_escape_and_recovery(
                conn, snapshot, actions, baseline_candidate, report["a0_verdict"],
                collection_context, recovery_state, str(instruction), int(decision_step),
            )
            report.update(recovery)
            return report

        if report["a0_verdict"] == "safe":
            if not isinstance(report["a0_branch_summary"].get("contact_monitoring"), dict) or report["a0_branch_summary"]["contact_monitoring"].get("status") != "complete":
                report.update({
                    "status": "a0_contact_monitoring_unknown",
                    "a0_verdict": "unknown",
                    "repair_status": "skipped_a0_unknown",
                    "must_abort_episode": True,
                })
                add_failure("A0_contact_monitoring_was_not_complete")
                return report

            # Near-Far operates only on the unique object-to-object forbid pair
            # already present in the validated Safety IR and bound to this
            # active Geometry Inventory. Never infer a pair from object names.
            near_far_result = None
            pair_subject_id = None
            pair_obstacle_id = None
            forecast_started = False
            try:
                if not isinstance(snapshot.get("snapshot_state"), dict):
                    raise ValueError("snapshot_state_missing_for_near_far_restore_verification")
                binding_validation = collection_context.get("semantic_binding_validation")
                if not isinstance(binding_validation, dict) or binding_validation.get("valid") is not True:
                    raise ValueError("semantic_geometry_binding_validation_missing_or_invalid")
                from research.gs_opd.smoke.near_far_gap_smoke import (
                    _resolve_bound_geom_sets,
                    _select_forbid_pair,
                )
                from research.gs_opd.repair.near_far import (
                    forecast_near_far_risk,
                )

                object_rules = [
                    rule for rule in teacher.get("object_contact_rules", [])
                    if isinstance(rule, dict)
                    and rule.get("decision") == "forbid"
                    and rule.get("predicate") == "contact"
                    and rule.get("phase") == "any"
                ]
                pairs = {
                    (rule.get("subject_id"), rule.get("target_id"))
                    for rule in object_rules
                    if isinstance(rule.get("subject_id"), str)
                    and isinstance(rule.get("target_id"), str)
                }
                if len(pairs) != 1:
                    raise ValueError("near_far_requires_one_unambiguous_forbid_object_contact_pair")
                pair_subject_id, pair_obstacle_id = next(iter(pairs))
                _select_forbid_pair(teacher, pair_subject_id, pair_obstacle_id)
                task_binding = collection_context.get("task_object_binding")
                subject_binding = binding_validation.get("objects", {}).get(pair_subject_id)
                if (
                    not isinstance(task_binding, dict)
                    or task_binding.get("object_id") != pair_subject_id
                    or not isinstance(subject_binding, dict)
                    or task_binding.get("body_id") != subject_binding.get("body_id")
                ):
                    raise ValueError("near_far_subject_does_not_match_the_validated_task_object_binding")
                _, target_geom_ids, obstacle_geom_ids = _resolve_bound_geom_sets(
                    binding_validation,
                    geometry_inventory,
                    pair_subject_id,
                    pair_obstacle_id,
                )
                forecast_started = True
                near_far_result = forecast_near_far_risk(
                    conn,
                    snapshot,
                    actions,
                    target_geom_ids,
                    obstacle_geom_ids,
                    decision_step=int(decision_step),
                )
                report["near_far_geom_bindings"] = {
                    "subject_object_id": pair_subject_id,
                    "obstacle_object_id": pair_obstacle_id,
                    "target_geom_ids": target_geom_ids,
                    "obstacle_geom_ids": obstacle_geom_ids,
                    "source": "Safety IR pair, semantic binding, and active Geometry Inventory",
                }
                report["near_far_result"] = near_far_result
                report["near_far_status"] = near_far_result.get("status")
                report["near_far_risk_verdict"] = near_far_result.get("risk_verdict")
            except Exception as exc:
                if forecast_started:
                    report["main_state_restored"] = False
                near_far_result = {
                    "status": "unknown",
                    "risk_verdict": "unknown",
                    "risk_warning": None,
                    "unknown_reasons": [f"{type(exc).__name__}: {exc}"],
                    "snapshot_restore": {"before": None, "after": None},
                }
                report["near_far_result"] = near_far_result
                report["near_far_status"] = "unknown"
                report["near_far_risk_verdict"] = "unknown"

            restore = near_far_result.get("snapshot_restore") if isinstance(near_far_result, dict) else None
            restore_after = restore.get("after") if isinstance(restore, dict) else None
            if isinstance(restore_after, dict) and restore_after.get("matched") is not True:
                report["main_state_restored"] = False
            if not _near_far_result_is_complete(
                near_far_result,
                snapshot,
                actions,
                report["near_far_geom_bindings"].get("target_geom_ids") if isinstance(report.get("near_far_geom_bindings"), dict) else None,
                report["near_far_geom_bindings"].get("obstacle_geom_ids") if isinstance(report.get("near_far_geom_bindings"), dict) else None,
                int(decision_step),
            ):
                report.update({
                    "status": "near_far_unknown",
                    "a0_verdict": "unknown",
                    "repair_status": "skipped_near_far_unknown",
                    "must_abort_episode": True,
                })
                for reason in near_far_result.get("unknown_reasons", []):
                    add_failure(reason)
                if report["main_state_restored"] is not True:
                    report["status"] = "aborted_main_state_restore_failed"
                    add_failure("near_far_snapshot_restore_failed")
                if not report["failure_reasons"]:
                    add_failure("near_far_forecast_did_not_complete_with_resolved_verdict")
                return report

            if near_far_result.get("risk_verdict") == "no_warning":
                report.update({
                    "status": "a0_safe",
                    "repair_status": "skipped_a0_safe_no_near_far_warning",
                    "training_action_chunk": actions.tolist(),
                    "training_label_valid": True,
                    "label_source": "compiler_and_near_far_verified_safe_A0",
                })
                return report

            report["repair_attempted"] = True
            report["repair_status"] = "early_repair_searching"
            try:
                from research.gs_opd.repair.early_repair import run_early_repair_search
                from research.gs_opd.repair.near_far import action_chunk_fingerprint

                compiled_a0 = next(
                    item for item in compiled.get("candidates", [])
                    if isinstance(item, dict)
                    and item.get("candidate_id") == a0_candidate_id
                    and item.get("verdict") == "safe"
                )
                a0_physical_validation = {
                    "main_state_restored": True,
                    "snapshot_id": report["snapshot_id"],
                    "action_chunk_sha256": action_chunk_fingerprint(actions),
                    "branch_candidate": baseline_candidate,
                    "compiler_candidate": compiled_a0,
                }
                early = run_early_repair_search(
                    conn,
                    snapshot,
                    actions,
                    near_far_result,
                    teacher,
                    semantic_results,
                    collection_context["semantic_binding_validation"],
                    geometry_inventory,
                    gripper_inventory,
                    collection_context["repair_context"],
                    collection_context.get("task_object_binding"),
                    a0_physical_validation,
                    subject_object_id=pair_subject_id,
                    obstacle_object_id=pair_obstacle_id,
                )
            except Exception as exc:
                early = {
                    "status": "not_executable",
                    "executable": False,
                    "verification_complete": False,
                    "candidate_results_complete": False,
                    "candidate_validation_complete": False,
                    "qualified_candidates": [],
                    "candidate_results": [],
                    "snapshot_restore": {"before": None, "after": None},
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            report["early_repair_search"] = early
            report["early_repair_status"] = early.get("status")
            report["main_state_restored"] = (
                isinstance(early.get("snapshot_restore"), dict)
                and isinstance(early["snapshot_restore"].get("after"), dict)
                and early["snapshot_restore"]["after"].get("matched") is True
            )
            validated_candidates = _validated_early_repair_candidates(early, actions.shape)
            report["early_repair_candidate_gap_exclusions"] = validated_candidates.get(
                "rejected_candidates", []
            )
            if not validated_candidates.get("valid"):
                report.update({
                    "status": (
                        "aborted_main_state_restore_failed"
                        if report["main_state_restored"] is not True
                        else "early_repair_unavailable"
                    ),
                    "repair_status": "early_repair_failed",
                    "must_abort_episode": True,
                })
                report["repair_failure_reason"] = validated_candidates.get("reason")
                add_failure(validated_candidates.get("reason"))
                for reason in early.get("errors", []):
                    add_failure(reason)
                return report

            eligible = sorted(
                validated_candidates["candidates"],
                key=lambda item: (-item["endpoint_gap_m"], item["candidate_id"]),
            )
            selected = eligible[0]
            plus = selected["actions"]
            endpoint_gaps = [item["endpoint_gap_m"] for item in eligible]
            selected_id = selected["candidate_id"]
            report.update({
                "status": "a0_safe_early_repaired",
                "repair_status": "early_repair_candidate_selected",
                "early_repair_status": "qualified_candidate_selected",
                "early_repair_selected_candidate_id": selected_id,
                "selected_candidate_id": selected_id,
                "selected_endpoint_gap_m": selected["endpoint_gap_m"],
                "baseline_a0_endpoint_gap_m": selected["baseline_a0_endpoint_gap_m"],
                "selected_gap_improvement_m": selected["gap_improvement_m"],
                "qualified_candidates_min_endpoint_gap_m": min(endpoint_gaps),
                "qualified_candidates_max_endpoint_gap_m": max(endpoint_gaps),
                "qualified_candidate_endpoint_gap_count": len(eligible),
                "repaired_action_chunk_A_plus": plus.tolist(),
                "training_action_chunk": plus.tolist(),
                "training_label_valid": True,
                "label_source": "near_far_verified_geometry_guided_A_plus",
            })
            return report

        if report["a0_verdict"] != "violation":
            report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
            for reason in compiled.get("unknown_reasons", []):
                add_failure(reason)
            for item in compiled.get("candidates", []):
                if isinstance(item, dict):
                    add_failure(item.get("reason"))
            if not report["failure_reasons"]:
                add_failure("A0_compiler_verdict_is_unknown")
            return report

        report["repair_attempted"] = True
        last_restore_state = {"restored": True}

        def evaluate_candidates(specifications):
            last_restore_state["restored"] = False
            branch = bridge_request({
                "op": "branch_rollout",
                "snapshot_id": report["snapshot_id"],
                "candidates": [
                    {"candidate_id": item["candidate_id"], "actions": item["actions"]}
                    for item in specifications
                ],
            })
            last_restore_state["restored"] = branch.get("main_state_restored") is True
            compiled_candidates = compile_forbid_contact(
                teacher,
                object_id,
                semantic_result,
                branch,
                geometry_inventory,
                gripper_inventory,
                object_semantic_results=semantic_results,
            )
            branch_by_id = {
                item.get("candidate_id"): item
                for item in branch.get("candidates", [])
                if isinstance(item, dict)
            }
            compiled_by_id = {
                item.get("candidate_id"): item
                for item in compiled_candidates.get("candidates", [])
                if isinstance(item, dict)
            }
            results = []
            for item in specifications:
                candidate_id = item["candidate_id"]
                candidate_branch = branch_by_id.get(candidate_id)
                candidate_compiled = compiled_by_id.get(candidate_id)
                results.append({
                    "candidate_id": candidate_id,
                    "branch_result": {
                        **branch,
                        "candidates": [candidate_branch] if candidate_branch is not None else [],
                    },
                    "compiled_result": {
                        **compiled_candidates,
                        "candidates": [candidate_compiled] if candidate_compiled is not None else [],
                    },
                })
            return results

        try:
            repair = run_verifier_guided_repair(
                actions,
                baseline_branch,
                compiled,
                collection_context["repair_context"],
                evaluate_candidates,
                task_object_binding=collection_context.get("task_object_binding"),
            )
        except Exception as exc:
            repair = {
                "status": "repair_exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "repaired_action_chunk_A_plus": None,
                "supervision": {"training_label_valid": False},
            }
        report["main_state_restored"] = last_restore_state["restored"]
        report["repair_search"] = _compact_repair_report(repair)
        report["repair_status"] = repair.get("status", "unknown")
        report["repair_failure_reason"] = repair.get("failure_reason")
        report["repaired_action_chunk_A_plus"] = repair.get("repaired_action_chunk_A_plus")
        plus = report["repaired_action_chunk_A_plus"]
        try:
            plus_array = np.asarray(plus, dtype=np.float32)
        except (TypeError, ValueError):
            plus_array = np.empty((0, 0), dtype=np.float32)
        valid_repair_label = (
            report["repair_status"] == "repaired_candidate_found"
            and repair.get("supervision", {}).get("training_label_valid") is True
            and isinstance(plus, list)
            and plus_array.shape == actions.shape
            and np.isfinite(plus_array).all()
        )
        if valid_repair_label:
            report["training_action_chunk"] = plus
            report["training_label_valid"] = True
            report["label_source"] = "compiler_verified_repaired_A_plus"
            report["status"] = "violation_repaired"
        else:
            report["status"] = (
                "aborted_main_state_restore_failed"
                if report["main_state_restored"] is not True
                else "violation_unrepaired"
            )
            add_failure(report["repair_failure_reason"])
            for reason in repair.get("rejected_reasons", []):
                add_failure(reason)
            if not report["failure_reasons"]:
                add_failure(report["repair_status"])
        if report["main_state_restored"] is not True:
            report["status"] = "aborted_main_state_restore_failed"
            report["must_abort_episode"] = True
            report["training_action_chunk"] = None
            report["training_label_valid"] = False
            report["label_source"] = None
            add_failure("repair_branch_did_not_restore_the_decision_state")
        if report["status"] == "aborted_main_state_restore_failed":
            report["must_abort_episode"] = True
        return report
    except (ConnectionError, OSError, EOFError) as exc:
        report.update({
            "status": "aborted_main_state_restore_failed",
            "main_state_restored": False,
            "must_abort_episode": True,
        })
        add_failure(f"Safety_Bridge_request_failed:{type(exc).__name__}:{exc}")
        return report
    except Exception as exc:
        report.update({"status": "a0_unknown", "repair_status": "skipped_a0_unknown"})
        # A parse/compiler-side failure does not itself mutate MuJoCo, but if a
        # branch had started and its restore was unconfirmed, fail closed.
        if report["main_state_restored"] is None:
            report["main_state_restored"] = False
            report["must_abort_episode"] = True
        add_failure(f"{type(exc).__name__}:{exc}")
        return report


def _new_safety_episode_summary():
    return {
        "a0_verdict_counts": {"safe": 0, "violation": 0, "unknown": 0},
        "repair_attempts": 0,
        "repair_status_counts": {},
        "valid_a_plus_count": 0,
        "valid_training_label_count": 0,
        "failure_reason_counts": {},
        "candidate_rejection_reason_counts": {},
    }


def run_on_policy_safety_collection_decision(
    conn, chunk, instruction, chunk_index, decision_step, collection_context,
):
    """Use one episode-local recovery state for the bound object-contact task."""
    from research.gs_opd.repair.local_recovery import RecoveryState

    state = None
    if collection_context["teacher"].get("object_contact_rules"):
        state = collection_context.setdefault("recovery_state", RecoveryState())
        if not isinstance(state, RecoveryState):
            raise ValueError("collection recovery_state must be episode-local RecoveryState")
    return _run_bias_collection_decision(
        conn, chunk, instruction, chunk_index, decision_step, collection_context,
        recovery_state=state,
    )


def select_verified_execution_chunk(decision, predicted_actions):
    """Select A0 or A+ only when its exact chunk passed the required gates."""
    try:
        a0 = np.asarray(predicted_actions, dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "student_A0_is_not_numeric"}
    if a0.ndim != 2 or a0.shape[1:] != (7,) or not len(a0):
        return {"valid": False, "reason": "student_A0_must_have_shape_(T,7)"}
    if not np.isfinite(a0).all():
        return {"valid": False, "reason": "student_A0_contains_non_finite_values"}
    if not isinstance(decision, dict) or decision.get("main_state_restored") is not True:
        return {"valid": False, "reason": "snapshot_restore_was_not_confirmed"}
    if not (
        isinstance(decision.get("a0_branch_summary"), dict)
        and decision["a0_branch_summary"].get("full_action_chunk_executed") is True
    ):
        return {"valid": False, "reason": "A0_did_not_receive_full_chunk_validation"}
    try:
        recorded_a0 = np.asarray(decision.get("original_action_chunk_A0"), dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "safety_record_A0_is_not_numeric"}
    if recorded_a0.shape != a0.shape or not np.isfinite(recorded_a0).all() or not np.array_equal(recorded_a0, a0):
        return {"valid": False, "reason": "safety_record_A0_does_not_match_student_prediction"}

    if "recovery_search" in decision:
        from research.gs_opd.repair.local_recovery import (
            select_verified_recovery_execution,
        )

        return select_verified_recovery_execution(decision, a0)

    verdict = decision.get("a0_verdict")
    if verdict == "safe":
        if decision.get("status") == "a0_safe_early_repaired":
            if not (
                decision.get("training_label_valid") is True
                and decision.get("repair_attempted") is True
                and decision.get("repair_status") == "early_repair_candidate_selected"
                and decision.get("early_repair_status") == "qualified_candidate_selected"
                and decision.get("main_state_restored") is True
            ):
                return {"valid": False, "reason": "early_repair_selection_or_restore_is_not_valid"}
            validated = _validated_early_repair_candidates(
                decision.get("early_repair_search"), a0.shape
            )
            if not validated.get("valid"):
                return {"valid": False, "reason": validated.get("reason")}
            candidate_id = decision.get("early_repair_selected_candidate_id")
            selected = next(
                (item for item in validated["candidates"] if item["candidate_id"] == candidate_id),
                None,
            )
            try:
                plus = np.asarray(decision.get("repaired_action_chunk_A_plus"), dtype=np.float32)
                labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
            except (TypeError, ValueError):
                return {"valid": False, "reason": "early_repair_selected_actions_are_not_numeric"}
            if (
                selected is None
                or plus.shape != a0.shape
                or labeled.shape != a0.shape
                or not np.isfinite(plus).all()
                or not np.isfinite(labeled).all()
                or not np.array_equal(plus, labeled)
                or not np.array_equal(plus, selected["actions"])
            ):
                return {"valid": False, "reason": "early_repair_selected_candidate_actions_do_not_match"}
            return {
                "valid": True,
                "actions": plus.copy(),
                "source": "near_far_verified_geometry_guided_A_plus",
                "candidate_id": candidate_id,
                "reason": None,
            }
        if decision.get("status") != "a0_safe" or decision.get("training_label_valid") is not True:
            return {"valid": False, "reason": "safe_A0_is_not_marked_valid"}
        try:
            labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
        except (TypeError, ValueError):
            return {"valid": False, "reason": "safe_A0_label_is_not_numeric"}
        if labeled.shape != a0.shape or not np.isfinite(labeled).all() or not np.array_equal(labeled, a0):
            return {"valid": False, "reason": "safe_A0_label_does_not_match_student_prediction"}
        return {
            "valid": True,
            "actions": a0.copy(),
            "source": "compiler_verified_safe_A0",
            "candidate_id": None,
            "reason": None,
        }

    if verdict != "violation":
        reasons = decision.get("failure_reasons")
        reason = reasons[0] if isinstance(reasons, list) and reasons else "A0_verdict_is_not_safe_or_violation"
        return {"valid": False, "reason": reason}
    if not (
        decision.get("status") == "violation_repaired"
        and decision.get("repair_attempted") is True
        and decision.get("repair_status") == "repaired_candidate_found"
        and decision.get("training_label_valid") is True
    ):
        return {
            "valid": False,
            "reason": decision.get("repair_failure_reason") or decision.get("repair_status") or "A0_violation_has_no_validated_repair",
        }
    search = decision.get("repair_search")
    selected = search.get("selected_candidate") if isinstance(search, dict) else None
    candidate_id = search.get("selected_candidate_id") if isinstance(search, dict) else None
    supervision = search.get("supervision") if isinstance(search, dict) else None
    if not (
        isinstance(selected, dict)
        and selected.get("candidate_id") == candidate_id
        and selected.get("verdict") == "accepted"
        and selected.get("safety_verdict") == "safe"
        and isinstance(supervision, dict)
        and supervision.get("training_label_valid") is True
        and supervision.get("valid_action_steps") == len(a0)
        and isinstance(selected.get("execution_coverage"), dict)
        and selected["execution_coverage"].get("training_label_valid") is True
        and selected["execution_coverage"].get("validated_action_steps") == len(a0)
        and isinstance(selected.get("action_constraints"), dict)
        and selected["action_constraints"].get("status") == "pass"
        and isinstance(selected.get("task_progress"), dict)
        and selected["task_progress"].get("status") == "pass"
    ):
        return {"valid": False, "reason": "selected_A_plus_lacks_full_validation"}
    try:
        plus = np.asarray(decision.get("repaired_action_chunk_A_plus"), dtype=np.float32)
        labeled = np.asarray(decision.get("training_action_chunk"), dtype=np.float32)
        candidate_actions = np.asarray(selected.get("actions"), dtype=np.float32)
    except (TypeError, ValueError):
        return {"valid": False, "reason": "validated_A_plus_is_not_numeric"}
    if any(value.shape != a0.shape or not np.isfinite(value).all() for value in (plus, labeled, candidate_actions)):
        return {"valid": False, "reason": "validated_A_plus_shape_or_values_are_invalid"}
    if not (np.array_equal(plus, labeled) and np.array_equal(plus, candidate_actions)):
        return {"valid": False, "reason": "A_plus_does_not_match_selected_validated_candidate"}
    if not isinstance(candidate_id, str) or not candidate_id:
        return {"valid": False, "reason": "selected_A_plus_candidate_id_is_missing"}
    return {
        "valid": True,
        "actions": plus.copy(),
        "source": "compiler_verified_repaired_A_plus",
        "candidate_id": candidate_id,
        "reason": None,
    }


def _classify_safety_abort(decision):
    """Classify a fail-closed stop using its actual safety/restore outcome."""
    if not isinstance(decision, dict) or decision.get("main_state_restored") is not True:
        return "main_state_restore_failed"
    status = decision.get("status")
    if status in ("invalid_A0_action_shape", "invalid_A0_action_values"):
        return "invalid_action"
    if status in ("local_recovery_unavailable", "actual_recovery_execution_failed", "bounded_recovery_stop"):
        return status
    if decision.get("near_far_status") == "unknown":
        return "near_far_verification_unknown"
    if decision.get("a0_verdict") == "unknown":
        return "safety_verification_unknown"
    if decision.get("early_repair_status") not in (None, "qualified_candidate_selected"):
        return "early_repair_failed"
    if status in ("violation_unrepaired", "repair_exception"):
        return "repair_failed"
    if decision.get("a0_verdict") == "violation":
        return "repair_validation_failed"
    return "action_validation_failed"


def _must_abort_safety_episode(decision):
    return (
        decision.get("must_abort_episode") is True
        or decision.get("main_state_restored") is not True
        or decision.get("execution_selection_valid") is False
    )


def _update_safety_episode_summary(summary, decision):
    verdict = decision.get("a0_verdict")
    if verdict not in summary["a0_verdict_counts"]:
        verdict = "unknown"
    summary["a0_verdict_counts"][verdict] += 1
    if decision.get("repair_attempted") is True:
        summary["repair_attempts"] += 1
        status = str(decision.get("repair_status") or "unknown")
        summary["repair_status_counts"][status] = summary["repair_status_counts"].get(status, 0) + 1
    if decision.get("repaired_action_chunk_A_plus") is not None and decision.get("training_label_valid") is True:
        summary["valid_a_plus_count"] += 1
    if decision.get("training_label_valid") is True:
        summary["valid_training_label_count"] += 1
    recovery = decision.get("recovery")
    if isinstance(recovery, dict):
        actual = recovery.get("actual_execution") or {}
        summary["recovery_decision_count"] = summary.get("recovery_decision_count", 0) + 1
        summary["verified_safe_recovery_chunk_count"] = summary.get("verified_safe_recovery_chunk_count", 0) + int(actual.get("verified_safe_recovery_chunk") is True)
        summary["task_success_observed_in_main_environment"] = summary.get("task_success_observed_in_main_environment", False) or actual.get("task_success_observed") is True
        if actual.get("status") == "complete":
            transitions = summary.setdefault("recovery_transition_counts", {})
            transition = recovery.get("transition", "unknown")
            transitions[transition] = transitions.get(transition, 0) + 1
        counts = summary.setdefault("recovery_rejection_reason_counts", {})
        search = decision.get("recovery_search") or {}
        for key in ("rejection_reason_counts", "constraint_rejection_reason_counts", "selection_rejection_reason_counts"):
            for reason, count in search.get(key, {}).items():
                counts[str(reason)] = counts.get(str(reason), 0) + count
    for reason in decision.get("failure_reasons", []):
        summary["failure_reason_counts"][str(reason)] = summary["failure_reason_counts"].get(str(reason), 0) + 1
    repair_search = decision.get("repair_search")
    attempts = repair_search.get("search_attempts", []) if isinstance(repair_search, dict) else []
    for item in attempts:
        if not isinstance(item, dict) or item.get("verdict") == "accepted":
            continue
        reason = item.get("reason")
        if reason is not None:
            reason = str(reason)
            summary["candidate_rejection_reason_counts"][reason] = (
                summary["candidate_rejection_reason_counts"].get(reason, 0) + 1
            )


def get_video_frame(data):
    image = data["images"]["agentview_image"]
    frame = np.frombuffer(image["data"], dtype=image["dtype"])
    frame = frame.reshape(image["shape"]).copy()
    return np.ascontiguousarray(get_libero_image({"agentview_image": frame}))


args = parse_args()

if args.seed is not None:
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

ckpt = args.checkpoint.expanduser().resolve()
ckpt_file = resolve_model_safetensors(ckpt) or resolve_full_weights(ckpt)
assert ckpt_file is not None, f"Checkpoint missing: {ckpt}"

with initialize_config_dir(config_dir="/workspace/zy_workspace/RLinf/evaluations/libero", version_base=None):
    cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")

cfg.rollout.model.model_path = str(ckpt)
OmegaConf.update(cfg.rollout.model, "openpi.torch_compile", False, force_add=True)
if args.lora_checkpoint is not None:
    if not args.lora_checkpoint.is_file():
        raise FileNotFoundError(f"LoRA adapter checkpoint missing: {args.lora_checkpoint}")
    OmegaConf.update(cfg.rollout.model, "is_lora", False, force_add=True)
    OmegaConf.update(cfg.rollout.model, "lora_rank", 32, force_add=True)
    OmegaConf.update(
        cfg.rollout.model,
        "openpi.action_expert_variant",
        "gemma_300m_lora",
        force_add=True,
    )
    OmegaConf.update(cfg.rollout.model, "openpi.config_name", "pi05_libero", force_add=True)
    OmegaConf.update(cfg.rollout.model, "openpi.action_chunk", 10, force_add=True)
    OmegaConf.update(cfg.rollout.model, "openpi.action_env_dim", 7, force_add=True)
    OmegaConf.update(cfg.rollout.model, "openpi.model_action_dim", 32, force_add=True)
norm_stats_path = Path(
    "/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft/physical-intelligence/libero/norm_stats.json"
)
OmegaConf.update(
    cfg.rollout.model,
    "openpi_data.norm_stats_path",
    str(norm_stats_path),
    force_add=True,
)

print("MODEL PATH:", ckpt, flush=True)
print("CHECKPOINT FILE:", ckpt_file, flush=True)
print("NORM STATS:", norm_stats_path, "exists=", norm_stats_path.is_file(), flush=True)
print("LOADING PI0.5...", flush=True)

model = get_model(cfg.rollout.model)
adapter_metadata = None
if args.lora_checkpoint is not None:
    from research.gs_opd.data.lora_checkpoint import load_lora_adapter

    adapter_metadata = load_lora_adapter(model, args.lora_checkpoint)
model = model.to("cuda:0").eval()
if adapter_metadata is not None:
    print("LORA ADAPTER LOADED:", json.dumps(adapter_metadata, ensure_ascii=False), flush=True)
print("MODEL LOADED", flush=True)

import time
import json

conn = Client(("127.0.0.1", 38567), authkey=b"gs-opd-local")
steps = 0
chunks = 0
episode_reward_sum = 0.0
success = False
done = False
start = time.monotonic()

try:
    conn.send({"op": "reset"})
    data = conn.recv()
    trace_writer = None
    collection_enabled = args.on_policy_safety_collection
    trace_output_path = args.on_policy_safety_out if collection_enabled else args.trace_out
    if trace_output_path is not None:
        import sys

        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from research.gs_opd.data.trajectory_writer import TrajectoryWriter

        trace_metadata = {
            "suite": "obstacle_avoidance",
            "level": 0,
            "task_id": 0,
            "instruction": data["instruction"],
            "checkpoint_path": str(ckpt),
            "checkpoint_file": str(ckpt_file),
            "lora_checkpoint_path": (
                str(args.lora_checkpoint) if args.lora_checkpoint is not None else None
            ),
            "physical_state_recording": bool(args.record_physical_state),
            "substep_contact_recording": bool(args.record_substep_contacts),
            "seed": args.seed,
            "max_steps": args.max_steps,
            "camera_resolution": 256,
            "env_seed": 0,
            "initial_state_index": 0,
        }
        if args.grasp_only_smoke:
            active_scene = data.get("scene_metadata", {})
            trace_metadata.update({
                "suite": active_scene.get("benchmark"),
                "level": active_scene.get("level"),
                "task_id": active_scene.get("task_id"),
                "camera_resolution": active_scene.get("camera_resolution"),
                "env_seed": active_scene.get("environment_seed"),
                "initial_state_index": active_scene.get("init_state_index"),
                "experiment_mode": "grasp_only_smoke",
                "grasp_smoke_object_id": args.grasp_smoke_object_id,
                "teacher_json_path": str(args.teacher_json.resolve()),
                "semantic_results_json_path": str(args.semantic_results_json.resolve()),
                "training_data": False,
                "model_parameter_source": "original_pi05_rlinf_libero_sft_base_only_no_lora",
            })
        if collection_enabled:
            trace_metadata.update({
                "collection_mode": "continuous_on_policy_safety",
                "teacher_json_path": str(args.teacher_json),
                "safety_object_id": args.object_id,
                "semantic_result_path": (
                    str(args.semantic_result) if args.semantic_result is not None else None
                ),
                "semantic_results_json_path": (
                    str(args.semantic_results_json)
                    if args.semantic_results_json is not None
                    else None
                ),
                "task_object_id": args.task_object_id,
                "task_object_semantic_result_path": (
                    str(args.task_object_semantic_result)
                    if args.task_object_semantic_result is not None
                    else None
                ),
                "task_binding_policy": "per-run Safety IR binding validated against the active fixed-task MuJoCo model",
            })
        trace_writer = TrajectoryWriter(
            trace_output_path,
            trace_metadata,
        )
    safety_episode_summary = _new_safety_episode_summary() if collection_enabled else None
    collection_context = None
    collection_preflight_failure = None
    collection_abort_reason = None
    collection_abort_status = None
    if collection_enabled:
        try:
            collection_context = prepare_on_policy_safety_collection(
                conn,
                args.teacher_json,
                args.object_id,
                args.semantic_result,
                task_object_id=args.task_object_id,
                task_object_semantic_result_path=args.task_object_semantic_result,
                semantic_results_path=args.semantic_results_json,
            )
            if trace_writer is not None:
                trace_writer.metadata["task_object_id"] = collection_context.get("task_object_id")
                trace_writer.metadata["semantic_binding_validation"] = collection_context.get(
                    "semantic_binding_validation"
                )
        except Exception as exc:
            collection_preflight_failure = f"{type(exc).__name__}:{exc}"
            safety_episode_summary["failure_reason_counts"][collection_preflight_failure] = 1
            print("SAFETY COLLECTION PREFLIGHT FAILED:", collection_preflight_failure, flush=True)
    grasp_smoke_context = None
    grasp_smoke_monitor = None
    grasp_smoke_preflight_failure = None
    grasp_smoke_initial_snapshot = None
    if args.grasp_only_smoke:
        def grasp_smoke_bridge_request(payload):
            conn.send(payload)
            response = conn.recv()
            if not isinstance(response, dict):
                raise RuntimeError("Safety Bridge returned a non-object response")
            return response

        try:
            from research.gs_opd.repair.trajectory_bundle_acceptance import (
                GraspStabilityMonitor,
                query_live_grasp,
            )

            expected_instruction = (
                "pick up the book and place it in the back compartment of the caddy"
            )
            scene = data.get("scene_metadata", {})
            if (
                scene.get("benchmark") != "obstacle_avoidance"
                or scene.get("level") != 0
                or scene.get("task_id") != 1
                or scene.get("init_state_index") != 0
                or scene.get("environment_seed") != 0
                or data.get("instruction") != expected_instruction
            ):
                raise ValueError("bridge_reset_did_not_return_obstacle_avoidance_L0T1_state0_seed0")

            teacher = json.loads(args.teacher_json.read_text(encoding="utf-8"))
            semantic_results = json.loads(
                args.semantic_results_json.read_text(encoding="utf-8")
            )
            semantic = semantic_results.get(args.grasp_smoke_object_id)
            teacher_object = next(
                (
                    item for item in teacher.get("objects", [])
                    if item.get("id") == args.grasp_smoke_object_id
                ),
                None,
            )
            decision = semantic.get("semantic_decision") if isinstance(semantic, dict) else None
            association = semantic.get("physical_association") if isinstance(semantic, dict) else None
            if (
                teacher_object is None
                or teacher_object.get("role") != "target_object"
                or not isinstance(decision, dict)
                or decision.get("object_id") != args.grasp_smoke_object_id
                or decision.get("status") != "resolved"
                or not str(semantic.get("review_status", "")).startswith("reviewed_")
                or not isinstance(association, dict)
                or association.get("status") != "matched_by_iou"
                or association.get("body_id") != 23
                or association.get("body_name") != "black_book_1_main"
                or association.get("geom_ids") != [91]
                or association.get("geom_names") != ["black_book_1_g0"]
            ):
                raise ValueError("reviewed_object_1_to_Body23_Geom91_binding_is_missing_or_inconsistent")

            inventory = grasp_smoke_bridge_request({"op": "geometry_inventory"})
            if (
                inventory.get("status") != "complete"
                or inventory.get("robot_classification_status") != "available"
            ):
                raise ValueError("active_bridge_geometry_inventory_is_incomplete")
            active_geom = next(
                (
                    item for item in inventory.get("geometries", [])
                    if item.get("geom_id") == 91
                ),
                None,
            )
            if (
                active_geom is None
                or active_geom.get("body_id") != 23
                or active_geom.get("body_name") != "black_book_1_main"
                or active_geom.get("geom_name") != "black_book_1_g0"
                or active_geom.get("category") != "scene"
            ):
                raise ValueError("reviewed_book_binding_disagrees_with_active_bridge_geometry")

            binding = {
                "object_id": args.grasp_smoke_object_id,
                "body_id": association["body_id"],
                "body_name": association["body_name"],
                "geom_ids": association["geom_ids"],
                "geom_names": association["geom_names"],
            }
            grasp_smoke_context = {
                "subject_id": args.grasp_smoke_object_id,
                "instruction": expected_instruction,
                "query_payload": {"carried_binding": binding},
                "review_status": semantic["review_status"],
                "binding_validation": {
                    "status": "pass",
                    "source": str(args.semantic_results_json.resolve()),
                    "active_geometry_inventory_status": inventory.get("status"),
                    "active_geom": active_geom,
                    "current_visual_reidentification_performed": False,
                },
            }
            initial_observation = grasp_smoke_bridge_request(
                {"op": "observe", "include_snapshot_state": True}
            )
            if initial_observation.get("error"):
                raise ValueError(f"initial_observation_failed:{initial_observation['error']}")
            if (
                initial_observation.get("instruction") != expected_instruction
                or not isinstance(initial_observation.get("snapshot_state"), dict)
            ):
                raise ValueError("initial_observation_or_physical_state_snapshot_is_incomplete")
            data = initial_observation
            grasp_smoke_initial_snapshot = initial_observation["snapshot_state"]
            grasp_smoke_monitor = GraspStabilityMonitor(required_samples=3)
            try:
                initial_grasp_state = query_live_grasp(conn, grasp_smoke_context)
            except Exception as exc:
                initial_grasp_state = {
                    "status": "unavailable",
                    "grasp_confirmed": False,
                    "physical_state_unchanged": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            grasp_smoke_monitor.update(initial_grasp_state, rollout_step=steps)
            print(
                "GRASP-ONLY SMOKE PREFLIGHT:",
                json.dumps(
                    {
                        "scene": scene,
                        "target_binding": binding,
                        "review_status": semantic["review_status"],
                        "active_geom": active_geom,
                        "initial_grasp_status": initial_grasp_state.get("status"),
                        "initial_grasp_reason": initial_grasp_state.get("reason"),
                        "required_distinct_states": grasp_smoke_monitor.required_samples,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            grasp_smoke_preflight_failure = f"{type(exc).__name__}: {exc}"
            print("GRASP-ONLY SMOKE PREFLIGHT FAILED:", grasp_smoke_preflight_failure, flush=True)
    trajectory_validation_context = None
    trajectory_grasp_monitor = None
    trajectory_validation_report = None
    trajectory_validation_attempted = False
    trajectory_validation_restore_abort = False
    direct_execution_stop = False
    trajectory_validation_preflight_failure = None
    if args.trajectory_bundle_physical_validation:
        try:
            import sys

            repo_root = str(Path(__file__).resolve().parents[3])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from research.gs_opd.repair.trajectory_bundle_acceptance import (
                GraspStabilityMonitor,
                prepare_live_acceptance_context,
                query_live_grasp,
            )
            from research.gs_opd.teacher.schema_v2 import parse_teacher_output_v2

            validation_teacher = parse_teacher_output_v2(
                args.teacher_json.read_text(encoding="utf-8")
            )
            validation_semantics = json.loads(
                args.semantic_results_json.read_text(encoding="utf-8")
            )
            trajectory_validation_context = prepare_live_acceptance_context(
                conn,
                validation_teacher,
                validation_semantics,
                data["instruction"],
                subject_id=args.trajectory_subject_object_id,
            )
            trajectory_validation_context["teacher_json_path"] = str(
                args.teacher_json.resolve()
            )
            trajectory_validation_context["semantic_results_json_path"] = str(
                args.semantic_results_json.resolve()
            )
            trajectory_grasp_monitor = GraspStabilityMonitor()
            initial_grasp_state = query_live_grasp(conn, trajectory_validation_context)
            trajectory_grasp_monitor.update(initial_grasp_state, rollout_step=steps)
            print(
                "TRAJECTORY VALIDATION SEARCHING FOR LIVE STABLE GRASP:",
                json.dumps(
                    {
                        "subject_object_id": trajectory_validation_context["subject_id"],
                        "required_distinct_states": trajectory_grasp_monitor.required_samples,
                        "initial_grasp_status": initial_grasp_state.get("status"),
                        "initial_grasp_reason": initial_grasp_state.get("reason"),
                        "bridge_connection": "same persistent connection used for Student step and Snapshot branches",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            trajectory_validation_preflight_failure = (
                f"{type(exc).__name__}:{exc}"
            )
            print(
                "TRAJECTORY VALIDATION PREFLIGHT BLOCKED:",
                trajectory_validation_preflight_failure,
                flush=True,
            )
    video_frames = [] if args.video_out is not None else None
    if video_frames is not None:
        video_frames.append(get_video_frame(data))
    print("TASK:", data["instruction"], flush=True)
    contact_probe_result = None

    while (
        steps < args.max_steps
        and not done
        and not success
        and collection_preflight_failure is None
        and collection_abort_reason is None
        and not trajectory_validation_restore_abort
        and not direct_execution_stop
        and grasp_smoke_preflight_failure is None
        and not (
            args.grasp_only_smoke
            and grasp_smoke_monitor is not None
            and grasp_smoke_monitor.stable
        )
    ):
        if (
            args.trajectory_bundle_physical_validation
            and not trajectory_validation_attempted
            and trajectory_validation_preflight_failure is None
            and trajectory_validation_context is not None
            and trajectory_grasp_monitor is not None
            and trajectory_grasp_monitor.stable
        ):
            trajectory_validation_attempted = True
            direct_latest_data = data
            try:
                if args.mink_osc_left:
                    from research.gs_opd.smoke.mink_osc_snapshot_smoke import (
                        run_live_snapshot_mink_osc_smoke,
                    )

                    trajectory_validation_report = run_live_snapshot_mink_osc_smoke(
                        conn,
                        trajectory_validation_context,
                        trajectory_grasp_monitor.samples[-1],
                        decision_step=steps,
                        grasp_stability=trajectory_grasp_monitor.diagnostics(),
                        output_json=args.trajectory_validation_out,
                        mink_python=args.mink_python,
                        model_binary=args.mink_model_binary,
                        model_metadata=args.mink_model_metadata,
                    )
                elif args.direct_bspline_osc_left:
                    from research.gs_opd.smoke.direct_bspline_osc_smoke import (
                        run_live_direct_bspline_osc_smoke,
                    )

                    trajectory_validation_report, direct_latest_data = run_live_direct_bspline_osc_smoke(
                        conn,
                        trajectory_validation_context,
                        trajectory_grasp_monitor.samples[-1],
                        decision_step=steps,
                        grasp_stability=trajectory_grasp_monitor.diagnostics(),
                        output_json=args.trajectory_validation_out,
                        trace_writer=trace_writer,
                        chunk_index=chunks,
                        step_index=steps,
                        initial_bridge_data=data,
                        video_frames=video_frames,
                        video_frame_encoder=get_video_frame,
                        candidate_id=args.bspline_candidate_id,
                        interpolate_left_001_to_004_fraction=(
                            args.bspline_left_001_toward_004_fraction
                        ),
                    )
                else:
                    from research.gs_opd.repair.trajectory_bundle_acceptance import (
                        run_snapshot_trajectory_acceptance,
                    )

                    trajectory_validation_report = run_snapshot_trajectory_acceptance(
                        conn,
                        trajectory_validation_context,
                        trajectory_grasp_monitor.samples[-1],
                        args.trajectory_validation_out,
                        decision_step=steps,
                        grasp_stability=trajectory_grasp_monitor.diagnostics(),
                    )
            except Exception as exc:
                trajectory_validation_report = {
                    "operation": "one_shot_geometry_guided_trajectory_physical_acceptance",
                    "status": "blocked",
                    "decision_step": steps,
                    "failure_reasons": [f"{type(exc).__name__}:{exc}"],
                    "main_environment_actions_by_acceptance": 0,
                    "main_environment_candidate_execution": False,
                    "training_label_created": False,
                }
            print(
                "TRAJECTORY PHYSICAL ACCEPTANCE:",
                json.dumps(
                    {
                        "status": trajectory_validation_report.get("status"),
                        "candidate_outcome": trajectory_validation_report.get(
                            "candidate_outcome"
                        ),
                        "decision_step": steps,
                        "live_grasp": trajectory_validation_report.get(
                            "live_grasp_at_snapshot", {}
                        ).get("status"),
                        "geometric_candidates": trajectory_validation_report.get(
                            "candidate_count"
                        ),
                        "branches_executed": trajectory_validation_report.get(
                            "branch_candidate_count"
                        ),
                        "accepted_tracking_candidates": len(
                            trajectory_validation_report.get("tracking", {}).get(
                                "executable_safe_candidate_ids", []
                            )
                        ),
                        "restore_verified": trajectory_validation_report.get(
                            "main_state_restored"
                        ),
                        "failure_reasons": trajectory_validation_report.get(
                            "failure_reasons", []
                        ),
                        "candidate_rejection_reason_counts": trajectory_validation_report.get(
                            "candidate_rejection_reason_counts", {}
                        ),
                        "report": str(args.trajectory_validation_out),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            from research.gs_opd.repair.trajectory_bundle_acceptance import (
                must_stop_rollout_after_acceptance,
            )
            if args.direct_bspline_osc_left:
                direct_count = int((trajectory_validation_report or {}).get("real_execution_steps", 0))
                episode_reward_sum += sum(
                    float(item.get("reward", 0.0))
                    for item in (trajectory_validation_report or {}).get("steps", [])
                )
                steps += direct_count
                route_completed = bool(
                    trajectory_validation_report
                    and trajectory_validation_report.get("route_completed") is True
                )
                if route_completed and isinstance(direct_latest_data, dict):
                    data = direct_latest_data
                if trajectory_validation_report and trajectory_validation_report.get("steps"):
                    last_transition = trajectory_validation_report["steps"][-1]
                    success = success or bool(last_transition.get("reward", 0.0) > 0)
                    done = bool(last_transition.get("done", False))
                direct_execution_stop = not route_completed
                print(
                    "DIRECT B-SPLINE→OSC ROUTE:",
                    (trajectory_validation_report or {}).get("status"),
                    (trajectory_validation_report or {}).get("failure_reason"),
                    "REAL OSC STEPS:",
                    direct_count,
                    "STUDENT_CONTINUATION:",
                    bool(route_completed and not done and not success),
                    flush=True,
                )
                if not route_completed or done or success:
                    break
                continue
            if args.mink_osc_left or must_stop_rollout_after_acceptance(trajectory_validation_report):
                trajectory_validation_restore_abort = True
                print(
                    "TRAJECTORY VALIDATION STOPPED ROLLOUT BEFORE NEXT STUDENT ACTION:",
                    trajectory_validation_report.get("status"),
                    trajectory_validation_report.get("failure_reasons", []),
                    flush=True,
                )
                break
        obs = {
            k: np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).copy()
            for k, v in data["images"].items()
        }
        obs.update({
            k: np.asarray(v, dtype=np.float32)
            for k, v in data["state"].items()
        })

        state = np.concatenate([
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        ]).astype(np.float32)

        env_obs = {
            "main_images": torch.from_numpy(np.ascontiguousarray(get_libero_image(obs))).unsqueeze(0),
            "wrist_images": torch.from_numpy(np.ascontiguousarray(get_libero_wrist_image(obs))).unsqueeze(0),
            "states": torch.from_numpy(state).unsqueeze(0),
            "task_descriptions": [data["instruction"]],
        }

        with torch.inference_mode():
            actions, _ = model.predict_action_batch(env_obs, mode="eval")

        chunk = actions[0].detach().float().cpu().numpy()
        if not collection_enabled:
            assert chunk.ndim == 2 and chunk.shape[1] == 7
            assert np.isfinite(chunk).all()
        chunk_index = chunks
        if trace_writer is not None and not collection_enabled:
            trace_writer.add_decision(chunk_index, steps, data, chunk)
        chunks += 1

        if args.contact_safety_probe:
            contact_probe_result = run_first_chunk_contact_probe(
                conn=conn,
                chunk=chunk,
                instruction=data["instruction"],
                teacher_json_path=args.teacher_json,
                object_id=args.object_id,
                semantic_result_path=args.semantic_result,
                out_dir=args.probe_out_dir,
            )
            break

        if args.verifier_guided_repair:
            contact_probe_result = run_first_chunk_verifier_guided_repair(
                conn=conn,
                chunk=chunk,
                instruction=data["instruction"],
                teacher_json_path=args.teacher_json,
                object_id=args.object_id,
                semantic_result_path=args.semantic_result,
                out_dir=args.repair_out_dir,
                task_object_id=args.task_object_id,
                task_object_semantic_result_path=args.task_object_semantic_result,
            )
            break

        if collection_enabled:
            try:
                safety_decision = run_on_policy_safety_collection_decision(
                    conn=conn,
                    chunk=chunk,
                    instruction=data["instruction"],
                    chunk_index=chunk_index,
                    decision_step=steps,
                    collection_context=collection_context,
                )
            except Exception as exc:
                safety_decision = {
                    "chunk_index": chunk_index,
                    "decision_step": steps,
                    "instruction": data["instruction"],
                    "original_action_chunk_A0": chunk.tolist(),
                    "a0_verdict": "unknown",
                    "status": "aborted_main_state_restore_failed",
                    "main_state_restored": False,
                    "must_abort_episode": True,
                    "training_action_chunk": None,
                    "training_label_valid": False,
                    "failure_reasons": [f"collection_decision_exception:{type(exc).__name__}:{exc}"],
                }
            selection = select_verified_execution_chunk(safety_decision, chunk)
            safety_decision["execution_selection_valid"] = selection["valid"]
            safety_decision["selected_action_source"] = selection.get("source")
            safety_decision["selected_candidate_id"] = selection.get("candidate_id")
            safety_decision["selected_action_chunk"] = (
                selection["actions"].tolist() if selection["valid"] else None
            )
            safety_decision["execution_selection_failure_reason"] = selection.get("reason")
            if not selection["valid"]:
                safety_decision.setdefault("failure_reasons", []).append(selection["reason"])
                collection_abort_reason = selection["reason"] or "no_verified_action_chunk"
                collection_abort_status = _classify_safety_abort(safety_decision)
            else:
                execution_chunk = selection["actions"]
            if chunk.ndim == 2:
                trace_writer.add_decision(
                    chunk_index,
                    steps,
                    data,
                    chunk,
                    safety_record=safety_decision,
                )
            if _must_abort_safety_episode(safety_decision):
                _update_safety_episode_summary(safety_episode_summary, safety_decision)
                collection_abort_reason = collection_abort_reason or safety_decision.get("status") or "branch_restore_unconfirmed"
                collection_abort_status = collection_abort_status or _classify_safety_abort(safety_decision)
                print(
                    "SAFETY COLLECTION ABORTED BEFORE MAIN ACTION:",
                    collection_abort_status,
                    collection_abort_reason,
                    flush=True,
                )
                break
        else:
            execution_chunk = chunk

        actual_chunk_actions = []
        actual_chunk_post = []
        for action_index, action in enumerate(execution_chunk):
            payload = {
                "op": "step_monitored" if args.record_substep_contacts else "step",
                "action": action.tolist(),
            }
            if args.record_substep_contacts:
                payload["action_index"] = steps
            if collection_enabled:
                payload["include_snapshot_state"] = True
            if args.record_physical_state:
                payload["include_snapshot_state"] = True
                payload["include_contacts"] = True
            conn.send(payload)
            response = conn.recv()
            if collection_enabled and (not isinstance(response, dict) or response.get("error")):
                collection_abort_reason = "main_environment_step_failed"
                collection_abort_status = "actual_recovery_execution_failed"
                safety_decision["training_label_valid"] = False
                safety_decision["must_abort_episode"] = True
                safety_decision["status"] = "actual_recovery_execution_failed"
                safety_decision["failure_reasons"].append(collection_abort_reason)
                break
            data = response
            steps += 1
            actual_chunk_actions.append(action.copy())
            actual_chunk_post.append(data)
            if (
                args.trajectory_bundle_physical_validation
                and not trajectory_validation_attempted
                and trajectory_validation_preflight_failure is None
                and trajectory_validation_context is not None
                and trajectory_grasp_monitor is not None
                and not data.get("done", False)
            ):
                try:
                    from research.gs_opd.repair.trajectory_bundle_acceptance import (
                        query_live_grasp,
                    )

                    grasp_state = query_live_grasp(conn, trajectory_validation_context)
                except Exception as exc:
                    grasp_state = {
                        "status": "unavailable",
                        "grasp_confirmed": False,
                        "physical_state_unchanged": False,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                trajectory_grasp_monitor.update(grasp_state, rollout_step=steps)
            if args.grasp_only_smoke and grasp_smoke_monitor is not None:
                try:
                    from research.gs_opd.repair.trajectory_bundle_acceptance import (
                        query_live_grasp,
                    )

                    grasp_state = query_live_grasp(conn, grasp_smoke_context)
                except Exception as exc:
                    grasp_state = {
                        "status": "unavailable",
                        "grasp_confirmed": False,
                        "physical_state_unchanged": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                grasp_smoke_monitor.update(grasp_state, rollout_step=steps)
            if trace_writer is not None:
                try:
                    trace_writer.add_step(
                        step_index=steps - 1, chunk_index=chunk_index,
                        action_index=action_index, action=action, bridge_data=data,
                    )
                except ValueError as exc:
                    if not collection_enabled:
                        raise
                    collection_abort_reason = str(exc)
                    collection_abort_status = "actual_recovery_execution_failed"
                    safety_decision["training_label_valid"] = False
                    safety_decision["must_abort_episode"] = True
                    safety_decision["failure_reasons"].append(str(exc))
                    break
            if video_frames is not None:
                video_frames.append(get_video_frame(data))
            success = success or data["reward"] > 0
            episode_reward_sum += float(data.get("reward", 0.0))
            done = bool(data["done"])

            if collection_enabled and "recovery_execution" in safety_decision:
                from research.gs_opd.repair.local_recovery import (
                    validate_actual_recovery_step,
                )

                checked = validate_actual_recovery_step(safety_decision, action_index, action, data)
                safety_decision["recovery"].setdefault("actual_step_checks", []).append(checked)
                if checked["status"] != "complete":
                    collection_abort_reason = checked["reason"]
                    collection_abort_status = "actual_recovery_execution_failed"
                    safety_decision.update({
                        "must_abort_episode": True, "training_label_valid": False,
                        "status": "actual_recovery_execution_failed",
                    })
                    safety_decision["failure_reasons"].append(collection_abort_reason)
                    break

            if steps % 20 == 0 or success or done:
                print(f"STEP={steps} REWARD={data['reward']} DONE={done} SUCCESS={success}", flush=True)

            if steps >= args.max_steps or done or success:
                break
            if (
                args.grasp_only_smoke
                and grasp_smoke_monitor is not None
                and grasp_smoke_monitor.stable
            ):
                print(
                    "STABLE GRASP DETECTED; STOPPING BEFORE THE NEXT STUDENT ACTION:",
                    json.dumps(grasp_smoke_monitor.diagnostics(), ensure_ascii=False),
                    flush=True,
                )
                break

        if collection_enabled:
            if "recovery_execution" in safety_decision:
                from research.gs_opd.repair.local_recovery import (
                    commit_recovery_execution,
                )
                from research.gs_opd.repair.near_far import query_geom_gap

                recovery_record = safety_decision["recovery"]
                actual_gap = query_geom_gap(
                    conn,
                    subject_geom_ids=recovery_record["subject_geom_ids"],
                    obstacle_geom_ids=recovery_record["obstacle_geom_ids"],
                    step_index=steps,
                )
                committed = commit_recovery_execution(
                    collection_context["recovery_state"], safety_decision,
                    actual_chunk_actions, actual_chunk_post, collection_context, actual_gap,
                )
                if committed.get("status") != "complete" or (committed.get("must_stop") and not success):
                    safety_decision.update({
                        "must_abort_episode": True, "training_label_valid": False,
                        "status": "actual_recovery_execution_failed" if committed.get("status") != "complete" else "bounded_recovery_stop",
                    })
                    reason = committed.get("reason", "bounded_recovery_progress_budget_exhausted")
                    safety_decision["failure_reasons"].append(reason)
                    collection_abort_reason = reason
                    collection_abort_status = safety_decision["status"]
            safety_decision = trace_writer.finalize_decision_execution(chunk_index, safety_decision)
            _update_safety_episode_summary(safety_episode_summary, safety_decision)
            if collection_abort_reason is not None:
                break

    if args.video_out is not None:
        from rlinf.envs.utils import save_rollout_video

        save_rollout_video(
            video_frames,
            str(args.video_out.parent),
            args.video_out.stem,
        )
        print("VIDEO:", args.video_out, flush=True)

    safety_summary_path = None
    if collection_enabled:
        from dataclasses import asdict

        if collection_preflight_failure is not None:
            episode_status = "collection_preflight_failed"
        elif collection_abort_reason is not None:
            episode_status = collection_abort_status or "safety_collection_stopped"
        elif success:
            episode_status = "success"
        elif done:
            episode_status = "environment_done"
        else:
            episode_status = "max_steps_reached"
        safety_episode_summary.update({
            "scope": collection_context["scope"] if collection_context is not None else {
                "suite": "obstacle_avoidance", "level": 0, "task_id": 0,
                "binding_scope": "preflight failed; no scene-general binding assumed",
            },
            "instruction": data.get("instruction"),
            "episode_status": episode_status,
            "steps": steps,
            "chunks": chunks,
            "decision_count": chunks,
            "preflight_failure": collection_preflight_failure,
            "abort_reason": collection_abort_reason,
            "data_path": str(trace_output_path),
            "task_success_episode": bool(success and collection_abort_reason is None and collection_preflight_failure is None),
            "high_quality_complete_demonstration": False,
            "recovery_state_final": (
                asdict(collection_context["recovery_state"])
                if collection_context is not None and "recovery_state" in collection_context else None
            ),
        })
        safety_summary_path = trace_output_path.with_suffix(".summary.json")
        safety_summary_path.parent.mkdir(parents=True, exist_ok=True)
        safety_summary_path.write_text(
            json.dumps(safety_episode_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trace_writer.metadata["safety_episode_summary"] = safety_episode_summary
        print(
            "SAFETY COLLECTION SUMMARY:",
            json.dumps(safety_episode_summary, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    trajectory_validation_summary_path = None
    if args.trajectory_bundle_physical_validation:
        from research.gs_opd.repair.trajectory_bundle_acceptance import _json_safe

        if trajectory_validation_report is None:
            if trajectory_validation_preflight_failure is not None:
                trajectory_status = "blocked"
                reasons = [trajectory_validation_preflight_failure]
            elif trajectory_grasp_monitor is not None and trajectory_grasp_monitor.stable:
                trajectory_status = "stable_grasp_found_without_following_decision_state"
                reasons = ["episode_ended_before_next_student_decision"]
            else:
                trajectory_status = "stable_grasp_not_found_within_episode"
                reasons = [
                    "episode_ended_before_consecutive_live_closed_gripper_and_opposing_finger_contact_evidence"
                ]
            trajectory_validation_report = {
                "operation": "one_shot_geometry_guided_trajectory_physical_acceptance",
                "status": trajectory_status,
                "failure_reasons": reasons,
                "main_environment_actions_by_acceptance": 0,
                "main_environment_candidate_execution": False,
                "training_label_created": False,
                "physical_safety_claim": False,
                "preflight_failure": trajectory_validation_preflight_failure,
                "binding_validation": (
                    trajectory_validation_context.get("binding_validation")
                    if trajectory_validation_context is not None
                    else None
                ),
            }
        trajectory_validation_report["episode_rollout"] = {
            "instruction": data.get("instruction"),
            "max_steps": args.max_steps,
            "executed_steps": steps,
            "decision_chunks": chunks,
            "reward_sum": float(episode_reward_sum),
            "success": bool(success),
            "done": bool(done),
            "episode_end_reason": (
                "direct_B_Spline_OSC_route_failed"
                if direct_execution_stop
                else
                "trajectory_validation_state_integrity_failure"
                if trajectory_validation_restore_abort
                else
                "success" if success else "environment_done" if done else "max_steps"
            ),
            "rollout_stopped_before_next_student_action": (
                trajectory_validation_restore_abort or direct_execution_stop
            ),
            "direct_route_student_continuation": {
                "route_completed": bool(trajectory_validation_report.get("route_completed")),
                "student_steps_after_route": max(
                    0,
                    steps
                    - int(trajectory_validation_report.get("decision_step", 0))
                    - int(trajectory_validation_report.get("real_execution_steps", 0)),
                ),
                "student_continued_from_latest_live_observation": bool(
                    trajectory_validation_report.get("route_completed")
                    and not direct_execution_stop
                    and steps
                    > int(trajectory_validation_report.get("decision_step", 0))
                    + int(trajectory_validation_report.get("real_execution_steps", 0))
                ),
            },
            "checkpoint_path": str(ckpt),
            "bridge_endpoint": "127.0.0.1:38567",
            "bridge_connection_policy": "the existing pi05_rollout Client connection was used for Student steps, grasp queries, Snapshot, geometry, and branch tracking",
        }
        if trajectory_grasp_monitor is not None:
            trajectory_validation_report["grasp_stability"] = (
                trajectory_grasp_monitor.diagnostics()
            )
        trajectory_validation_report["physical_safety_claim"] = False
        trajectory_validation_report["training_label_created"] = False
        trajectory_validation_report["main_environment_candidate_execution"] = bool(
            args.direct_bspline_osc_left
            and trajectory_validation_report.get("real_execution_steps", 0) > 0
        )
        trajectory_validation_report["main_environment_actions_by_acceptance"] = int(
            trajectory_validation_report.get("real_execution_steps", 0)
        )
        trajectory_validation_summary_path = args.trajectory_validation_out
        trajectory_validation_summary_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_validation_summary_path.write_text(
            json.dumps(_json_safe(trajectory_validation_report), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(
            "TRAJECTORY VALIDATION SUMMARY:",
            json.dumps(
                {
                    "status": trajectory_validation_report.get("status"),
                    "candidate_outcome": trajectory_validation_report.get(
                        "candidate_outcome"
                    ),
                    "steps": steps,
                    "stable_grasp_observed": (
                        trajectory_grasp_monitor.stable
                        if trajectory_grasp_monitor is not None
                        else False
                    ),
                    "branches_executed": trajectory_validation_report.get(
                        "branch_candidate_count", 0
                    ),
                    "restore_verified": trajectory_validation_report.get(
                        "main_state_restored"
                    ),
                    "failure_reasons": trajectory_validation_report.get(
                        "failure_reasons", []
                    ),
                    "candidate_rejection_reason_counts": trajectory_validation_report.get(
                        "candidate_rejection_reason_counts", {}
                    ),
                    "report": str(trajectory_validation_summary_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    grasp_smoke_report = None
    grasp_smoke_report_path = None
    if args.grasp_only_smoke:
        diagnostics = (
            grasp_smoke_monitor.diagnostics()
            if grasp_smoke_monitor is not None
            else {
                "required_distinct_consecutive_states": 3,
                "current_consecutive_states": 0,
                "stable_grasp_observed": False,
                "unique_state_samples": 0,
                "valid_grasp_samples": 0,
                "unavailable_sample_count": 0,
                "samples": [],
            }
        )
        stable = diagnostics.get("stable_grasp_observed") is True
        samples = diagnostics.get("samples", [])
        if stable:
            status = "stable_grasp_detected"
            failure_category = None
            stop_reason = "three_distinct_consecutive_closed_bilateral_contact_states"
        elif grasp_smoke_preflight_failure is not None or any(
            item.get("status") == "unavailable"
            or item.get("physical_state_unchanged") is not True
            or not isinstance(item.get("state_id"), str)
            or not item.get("state_id")
            for item in samples
        ):
            status = "state_detection_failed"
            failure_category = "state_detection_failure"
            stop_reason = grasp_smoke_preflight_failure or "live_grasp_state_evidence_unavailable_or_invalid"
        elif any(
            item.get("status") == "pass"
            and item.get("grasp_confirmed") is True
            and item.get("gripper_closed") is True
            and isinstance(item.get("finger_contacts"), dict)
            and item["finger_contacts"].get("left") is True
            and item["finger_contacts"].get("right") is True
            for item in samples
        ):
            status = "grasp_unstable"
            failure_category = "grasp_unstable"
            stop_reason = "stable_contact_evidence_did_not_reach_three_consecutive_distinct_states"
        elif any(
            item.get("gripper_closed") is True
            and isinstance(item.get("finger_contacts"), dict)
            and not (
                item["finger_contacts"].get("left") is True
                and item["finger_contacts"].get("right") is True
            )
            for item in samples
        ):
            status = "gripper_contact_insufficient"
            failure_category = "gripper_contact_insufficient"
            stop_reason = "closed_gripper_did_not_have_bilateral_opposing_finger_contacts"
        else:
            status = "book_not_grasped"
            failure_category = "model_did_not_grasp_book"
            stop_reason = (
                "task_ended_before_stable_grasp"
                if done or success
                else "200_action_budget_exhausted_without_stable_grasp"
                if steps >= args.max_steps
                else "rollout_stopped_without_stable_grasp"
            )
        grasp_smoke_report = {
            "operation": "L0T1_state0_raw_pi05_base_grasp_only_smoke",
            "status": status,
            "failure_category": failure_category,
            "stop_reason": stop_reason,
            "scene": data.get("scene_metadata"),
            "instruction": data.get("instruction"),
            "target_object_id": args.grasp_smoke_object_id,
            "review_status": (
                grasp_smoke_context.get("review_status")
                if grasp_smoke_context is not None
                else None
            ),
            "binding_validation": (
                grasp_smoke_context.get("binding_validation")
                if grasp_smoke_context is not None
                else None
            ),
            "model": {
                "checkpoint_path": str(ckpt),
                "checkpoint_file": str(ckpt_file),
                "lora_adapter_path": None,
                "parameter_source": "original_pi05_rlinf_libero_sft_base_only",
            },
            "action_source": "raw_pi05_student_predicted_action_chunks",
            "online_route_or_controller_handoff": False,
            "max_environment_actions": args.max_steps,
            "executed_environment_actions": steps,
            "decision_chunks": chunks,
            "stable_grasp": diagnostics,
            "task_reward_sum": float(episode_reward_sum),
            "task_success_once": bool(success),
            "environment_done": bool(done),
            "initial_physical_state_snapshot": grasp_smoke_initial_snapshot,
            "trace_npz": str(trace_output_path),
            "video_mp4": str(args.video_out),
            "training_data_created": False,
            "trajectory_safety_verdict": "not_evaluated",
        }
        grasp_smoke_report_path = args.result_out.with_name("grasp_stability_report.json")
        grasp_smoke_report_path.parent.mkdir(parents=True, exist_ok=True)
        grasp_smoke_report_path.write_text(
            json.dumps(grasp_smoke_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if trace_writer is not None:
            trace_writer.metadata["grasp_only_smoke_status"] = status
            trace_writer.metadata["grasp_stability_monitor"] = diagnostics
            trace_writer.metadata["training_data"] = False
        print(
            "GRASP-ONLY SMOKE SUMMARY:",
            json.dumps(
                {
                    "status": status,
                    "failure_category": failure_category,
                    "steps": steps,
                    "stable_grasp_observed": stable,
                    "reward_sum": episode_reward_sum,
                    "done": done,
                    "report": str(grasp_smoke_report_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if trace_writer is not None:
        trace_writer.save(steps=steps, chunks=chunks)
        print("TRACE:" if not collection_enabled else "SAFETY DATA:", trace_output_path, flush=True)
        if safety_summary_path is not None:
            print("SAFETY SUMMARY:", safety_summary_path, flush=True)

    result = {
        "suite": "obstacle_avoidance",
        "level": 0,
        "task_id": 0,
        "steps": steps,
        "chunks": chunks,
        "reward_sum": float(episode_reward_sum),
        "success_once": success,
        "done": done,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "camera_resolution": 256,
        "checkpoint_path": str(ckpt),
        "lora_checkpoint_path": (
            str(args.lora_checkpoint) if args.lora_checkpoint is not None else None
        ),
        "physical_state_recording": bool(args.record_physical_state),
        "substep_contact_recording": bool(args.record_substep_contacts),
        "seed": args.seed,
    }
    if args.grasp_only_smoke:
        scene = data.get("scene_metadata", {})
        result.update({
            "suite": scene.get("benchmark"),
            "level": scene.get("level"),
            "task_id": scene.get("task_id"),
            "camera_resolution": scene.get("camera_resolution"),
            "environment_seed": scene.get("environment_seed"),
            "init_state_index": scene.get("init_state_index"),
            "checkpoint_file": str(ckpt_file),
            "grasp_only_smoke": grasp_smoke_report,
            "grasp_stability_report_path": str(grasp_smoke_report_path),
        })
    if contact_probe_result is not None:
        if args.contact_safety_probe:
            result["contact_safety_probe"] = contact_probe_result["verdict"]
        elif args.verifier_guided_repair:
            result["verifier_guided_repair"] = contact_probe_result["status"]
    if collection_enabled:
        result["on_policy_safety_collection"] = safety_episode_summary
    if args.trajectory_bundle_physical_validation:
        result["trajectory_bundle_physical_validation"] = {
            "status": trajectory_validation_report.get("status"),
            "report_path": str(trajectory_validation_summary_path),
            "requested_candidate_id": trajectory_validation_report.get(
                "requested_candidate_id"
            ),
            "selected_candidate_id": trajectory_validation_report.get(
                "selected_candidate_id"
            ),
            "selected_route_direction": trajectory_validation_report.get(
                "selected_route_direction"
            ),
            "route_interpolation": trajectory_validation_report.get(
                "route_interpolation"
            ),
            "actual_action_source": trajectory_validation_report.get(
                "actual_action_source"
            ),
            "reference_speed_m_s": trajectory_validation_report.get(
                "reference_speed_m_s"
            ),
            "branch_candidate_count": trajectory_validation_report.get("branch_candidate_count", 0),
            "restore_verified": trajectory_validation_report.get("main_state_restored"),
            "real_execution_steps": trajectory_validation_report.get("real_execution_steps", 0),
            "route_completed": trajectory_validation_report.get("route_completed", False),
            "first_failure_reason": trajectory_validation_report.get("failure_reason"),
            "failure_reasons": trajectory_validation_report.get("failure_reasons", []),
        }

    args.result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_out, "w") as f:
        json.dump(result, f, indent=2)

    print("RESULT:", result, flush=True)
    print("===== PI05 SAFETY ROLLOUT FINISHED =====", flush=True)
finally:
    conn.close()
