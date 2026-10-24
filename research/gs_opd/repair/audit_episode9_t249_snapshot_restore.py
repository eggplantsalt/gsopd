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

"""Read-only, fail-closed restore probe for the archived Episode 9 t=249 state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.gs_opd.repair.automatic_recovery_runtime import BridgeProcess


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARCHIVE = Path(
    "/opt/data/private/gs_opd/rollouts/"
    "task_success_first_safety_reconstruction_l0t2_20261023_attempt1"
)
DEFAULT_PYTHON = Path("/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python")


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--bridge-port", type=int, default=38772)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    episode_dir = args.archive_root / "episode9"
    snapshot_path = episode_dir / "snapshots/main_timestep_000249.pkl"
    run_config = json.loads((episode_dir / "run_config.json").read_text())
    config_path = Path(run_config["config_path"])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text())
    payload = snapshot_path.read_bytes()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "gs_opd_episode9_t249_restore_probe.v1",
        "read_only": True,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": hashlib.sha256(payload).hexdigest(),
        "snapshot_bytes": len(payload),
        "expected_model_geometry_sha256": "1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752",
        "bridge_port": int(args.bridge_port),
        "demonstration_horizon": 1800,
        "physical_action_steps": 0,
        "physical_substeps": 0,
        "post_restore_observation_requested": False,
    }
    bridge = BridgeProcess(
        args.output_dir, python=args.python, port=args.bridge_port
    )
    try:
        bridge.start(
            benchmark=config["benchmark"],
            level=int(config["level"]),
            task_id=int(config["task_id"]),
            init_state=int(config["init_state"]),
            seed=int(config["environment_seed"]),
            demonstration_horizon=1800,
        )
        try:
            response = bridge.request(
                {"op": "restore_runtime_snapshot", "payload": payload}
            )
        except RuntimeError as exc:
            report["restore_status"] = "restore_failed"
            report["restore_error"] = str(exc)
            report["scene_and_model_validation"] = (
                "failed_at_model_geometry_signature"
                if "model geometry differs" in str(exc)
                else "failed"
            )
            report["restore_state"] = None
            report["post_restore_observation_requested"] = False
            (args.output_dir / "restore_probe.json").write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2
        report["restore_status"] = response.get("status")
        report["restore_error"] = response.get("error")
        report["scene_and_model_validation"] = (
            "failed_at_model_geometry_signature"
            if response.get("status") == "restore_failed"
            and "model geometry differs" in str(response.get("error", ""))
            else "passed" if response.get("status") == "restored" else "failed"
        )
        if response.get("status") != "restored":
            report["restore_state"] = None
            report["post_restore_observation_requested"] = False
            (args.output_dir / "restore_probe.json").write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2

        report["restore_state"] = response.get("snapshot_state")
        report["post_restore_observation_requested"] = True
        observation = bridge.request(
            {"op": "observe", "include_snapshot_state": True}
        )
        native = bridge.request({"op": "native_task_status"})
        exported = bridge.request({"op": "export_runtime_snapshot"})
        report["observed_timestep"] = observation.get("timestep")
        report["native_task_status"] = native
        report["reexport_status"] = exported.get("status")
        report["reexport_sha256"] = exported.get("payload_sha256")
        report["snapshot_payload_byte_identical_after_restore"] = (
            exported.get("payload") == payload
        )
        report["restore_state_matches_observation_timestep"] = (
            observation.get("timestep") == 249
        )
        (args.output_dir / "restore_probe.json").write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
            + "\n"
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=_json_default))
        return 0 if report["snapshot_payload_byte_identical_after_restore"] else 3
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
