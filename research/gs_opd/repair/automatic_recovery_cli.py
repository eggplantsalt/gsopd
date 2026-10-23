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

"""Launch one autonomous L0T2 recovery episode."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from research.gs_opd.repair.automatic_recovery import (
    AutomaticSafetyRecoveryEngine,
    RecoveryArtifactWriter,
)
from research.gs_opd.repair.automatic_recovery_runtime import (
    AutomaticRecoveryBridgeBackend,
    BridgeProcess,
    DEFAULT_ENV_PYTHON,
    DEFAULT_PI05_CHECKPOINT,
    load_runtime_config,
    sha256_file,
)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/l0t2_automatic_recovery.json"
DEFAULT_OUTPUT_ROOT = Path("/opt/data/private/gs_opd/rollouts")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic L0T2 Safety Recovery Engine")
    parser.add_argument(
        "--mode", required=True, choices=("t10-snapshot", "runtime-snapshot", "episode")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38569)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_PI05_CHECKPOINT)
    parser.add_argument("--student-device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    config_data, recovery_config = load_runtime_config(args.config)
    if args.output_dir is None:
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_root / f"automatic_safety_recovery_l0t2_{args.mode}_{timestamp}"
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
                "recovery_config": asdict(recovery_config),
                "student_checkpoint": str(args.student_checkpoint),
                "student_checkpoint_sha256": sha256_file(args.student_checkpoint / "model.safetensors"),
                "student_device": args.student_device,
                "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_artifacts": config_data["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    bridge = BridgeProcess(output_dir, python=args.python, port=args.bridge_port)
    backend = None
    started = time.monotonic()
    try:
        print("Loading the unchanged Pi0.5 Student checkpoint...", flush=True)
        from research.gs_opd.repair.automatic_recovery_runtime import Pi05StudentPolicy

        student = Pi05StudentPolicy(args.student_checkpoint, device=args.student_device)
        print("Starting the native LIBERO-Safety Bridge...", flush=True)
        bridge.start(
            benchmark=config_data["benchmark"],
            level=int(config_data["level"]),
            task_id=int(config_data["task_id"]),
            init_state=int(config_data["init_state"]),
            seed=int(config_data["environment_seed"]),
        )
        backend = AutomaticRecoveryBridgeBackend(
            bridge,
            config_data,
            recovery_config,
            output_dir,
            RecoveryArtifactWriter(output_dir),
            mode=args.mode,
            student=student,
            archived_trace_path=Path(config_data["artifacts"]["archived_student_trace"]),
            grasp_template_path=Path(config_data["artifacts"]["historical_grasp_template"]),
            runtime_snapshot_path=Path(config_data["artifacts"]["t10_runtime_snapshot"]),
            t10_verification_path=Path(config_data["artifacts"]["t10_verification"]),
        )
        start_state = backend.restore_start_state()
        (output_dir / "start_state.json").write_text(
            json.dumps(start_state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        context = backend.current_context()
        context.validate()
        backend.initialize_trajectory()
        artifacts = RecoveryArtifactWriter(output_dir)
        engine = AutomaticSafetyRecoveryEngine(backend, recovery_config, artifacts=artifacts)
        result = engine.run()
        backend.finalize(asdict(result))
        print(
            json.dumps(
                {
                    "status": result.status,
                    "state": result.state.value,
                    "local_recovery_success": result.local_recovery_success,
                    "complete_task_success": result.complete_task_success,
                    "complete_safety_success": result.complete_safety_success,
                    "standard_horizon_success": result.standard_horizon_success,
                    "steps_executed": result.steps_executed,
                    "replans": result.replans,
                    "reason": result.reason,
                    "phase": result.phase,
                    "elapsed_s": time.monotonic() - started,
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result.complete_task_success and result.complete_safety_success else 2
    except Exception as exc:
        failure = {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.monotonic() - started,
            "output_dir": str(output_dir),
        }
        (output_dir / "startup_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), flush=True)
        return 2
    finally:
        if backend is not None:
            for writer in backend._video_writers.values():
                writer.release()
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
