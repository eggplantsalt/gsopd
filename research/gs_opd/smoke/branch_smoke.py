"""Check action-branch isolation and main-state restoration in Safety Bridge."""
import argparse
import json
from multiprocessing.connection import Client
import sys
import numpy as np
AUTHKEY = b"gs-opd-local"
MAIN_ADVANCE_ACTIONS = np.asarray(
    [
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.04, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)
ACTION_A = np.asarray(
    [
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, -0.04, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)
ACTION_B = np.asarray(
    [
        [-0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, -0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.04, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)
RESTORE_PROBE_ACTION = np.asarray(
    [0.00, 0.00, -0.04, 0.00, 0.00, 0.00, 0.00], dtype=np.float32
)
PROBE_ACTIONS = np.stack([RESTORE_PROBE_ACTION, *ACTION_A[1:]])
DIAGNOSTIC_COMPONENTS = (
    "mujoco_integration_state",
    "robot_states",
    "object_states",
)


def receive_ok(connection, request):
    connection.send(request)
    result = connection.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def compare_detailed(left, right, path="state"):
    """Return the largest numeric difference and its exact nested location."""
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"{path} keys differ: {left.keys()} vs {right.keys()}")
        details = [
            compare_detailed(left[key], right[key], f"{path}.{key}")
            for key in left
        ]
        return max(details, key=lambda item: item[0], default=(0.0, path))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"{path} lengths differ: {len(left)} vs {len(right)}")
        details = [
            compare_detailed(a, b, f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(left, right))
        ]
        return max(details, key=lambda item: item[0], default=(0.0, path))
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        if left != right:
            raise ValueError(f"{path} differs: {left!r} vs {right!r}")
        return 0.0, path
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return abs(float(left) - float(right)), path
    if left != right:
        raise ValueError(f"{path} differs: {left!r} vs {right!r}")
    return 0.0, path


def compare_values(left, right, path="state"):
    """Check structure and non-numeric fields, returning maximum numeric error."""
    return compare_detailed(left, right, path)[0]


def report_state_errors(label, left, right):
    """Print per-component max errors and the exact field/index of each max."""
    errors = {}
    for component in DIAGNOSTIC_COMPONENTS:
        error, location = compare_detailed(
            left[component], right[component], component
        )
        errors[component] = (error, location)
        print(
            f"{label}.{component}: max_abs_error={error:.3e} "
            f"at {location}"
        )
    return errors


def max_state_detail(errors):
    _component, (error, location) = max(
        errors.items(), key=lambda item: item[1][0], default=("state", (0.0, "state"))
    )
    return error, location


def report_result_error(label, left, right):
    error, location = compare_detailed(left, right, label)
    print(f"{label}: max_abs_error={error:.3e} at {location}")
    return error


def flatten_numeric(value):
    if isinstance(value, dict):
        output = []
        for key in sorted(value):
            output.extend(flatten_numeric(value[key]))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(flatten_numeric(item))
        return output
    if isinstance(value, (bool, int, float, np.number)):
        return [float(value)]
    raise TypeError(f"Non-numeric physics diagnostic: {type(value).__name__}")


def component_delta_linf(start_state, step_state, key):
    start = np.asarray(flatten_numeric(start_state[key]), dtype=np.float64)
    current = np.asarray(flatten_numeric(step_state[key]), dtype=np.float64)
    if start.shape != current.shape:
        raise ValueError(f"{key} diagnostic shape changed within a branch")
    return float(np.max(np.abs(current - start))) if start.size else 0.0


def print_candidate(candidate, verbose_contacts=False):
    if "start_state" not in candidate:
        print(
            f"candidate {candidate['candidate_id']}: "
            f"no start state ({candidate.get('error')})"
        )
        return
    start_physics = np.asarray(
        candidate["start_state"]["mujoco_integration_state"], dtype=np.float64
    )
    print(
        f"candidate {candidate['candidate_id']}: status={candidate['status']}, "
        f"steps={candidate['executed_steps']}/{candidate['requested_steps']}"
    )
    for step in candidate["steps"]:
        physics = np.asarray(
            step["state"]["mujoco_integration_state"], dtype=np.float64
        )
        if physics.shape != start_physics.shape:
            raise ValueError("MuJoCo integration-state shape changed within a branch")
        delta = physics - start_physics
        delta_linf = float(np.max(np.abs(delta))) if delta.size else 0.0
        robot_delta = component_delta_linf(
            candidate["start_state"], step["state"], "robot_states"
        )
        object_delta = component_delta_linf(
            candidate["start_state"], step["state"], "object_states"
        )
        contacts = step["contacts"]
        print(
            f"  step={step['step_index']} reward={step['reward']:.6g} "
            f"done={step['done']} integration_delta_linf={delta_linf:.3e} "
            f"robot_delta_linf={robot_delta:.3e} "
            f"object_delta_linf={object_delta:.3e} "
            f"contact_status={contacts['status']} "
            f"contact_count={len(contacts['contacts'])}"
        )
        if verbose_contacts:
            print("    contacts=" + json.dumps(contacts, sort_keys=True))
    if candidate["status"] != "ok":
        print(f"  candidate error: {candidate.get('error')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Strict tolerance for saved MuJoCo state and branch isolation.",
    )
    parser.add_argument(
        "--replay-atol",
        type=float,
        default=1e-3,
        help="Exploratory tolerance for state/observation after action replay (default: 1e-3).",
    )
    parser.add_argument("--verbose-contacts", action="store_true")
    args = parser.parse_args()
    if args.atol < 0 or args.replay_atol < 0:
        parser.error("tolerances must be non-negative")
    try:
        connection = Client((args.host, args.port), authkey=AUTHKEY)
    except OSError as exc:
        print(f"FAIL: cannot connect to Safety Bridge at {args.host}:{args.port}: {exc}")
        return 1
    try:
        receive_ok(connection, {"op": "reset"})
        snapshot_s0 = receive_ok(connection, {"op": "snapshot"})
        snapshot_s0_id = snapshot_s0["snapshot_id"]
        state_s0 = snapshot_s0["snapshot_state"]
        main_steps = 0
        for action_index, action in enumerate(MAIN_ADVANCE_ACTIONS):
            if not np.any(action):
                raise AssertionError("S0-to-S1 main-environment actions must be non-zero")
            step_result = receive_ok(
                connection,
                {"op": "step", "action": action.tolist()},
            )
            main_steps += 1
            print(
                f"main advance step={action_index} reward={step_result['reward']:.6g} "
                f"done={step_result['done']}"
            )
            if step_result["done"]:
                print("Main environment terminated during S0-to-S1 advance.", flush=True)
                break
        snapshot_s1 = receive_ok(connection, {"op": "snapshot"})
        snapshot_s1_id = snapshot_s1["snapshot_id"]
        state_s1 = snapshot_s1["snapshot_state"]
        s0_s1_errors = report_state_errors("S0_vs_S1", state_s0, state_s1)
        s0_s1_delta, s0_s1_location = max_state_detail(s0_s1_errors)
        if not np.any(PROBE_ACTIONS):
            raise AssertionError("S1 replay-probe actions must be non-zero")

        def run_probe():
            records = []
            for action in PROBE_ACTIONS:
                reply = receive_ok(
                    connection,
                    {
                        "op": "step",
                        "action": action.tolist(),
                        "include_snapshot_state": True,
                    },
                )
                records.append({
                    "physics": reply["snapshot_state"],
                    "bridge_state": reply["state"],
                    "reward": reply["reward"],
                    "done": reply["done"],
                })
                if reply["done"]:
                    break
            return records

        original_continuation = run_probe()
        probe_errors = report_state_errors(
            "S1_vs_after_probe", state_s1, original_continuation[-1]["physics"]
        )
        probe_delta, probe_location = max_state_detail(probe_errors)

        direct_restore = receive_ok(
            connection, {"op": "restore", "snapshot_id": snapshot_s1_id}
        )
        state_after_direct_restore = direct_restore["snapshot_state"]
        direct_restore_errors = report_state_errors(
            "S1_vs_direct_restore", state_s1, state_after_direct_restore
        )
        direct_restore_error, direct_restore_location = max_state_detail(
            direct_restore_errors
        )
        restored_physics_error = direct_restore_errors["mujoco_integration_state"][0]

        restored_continuation = run_probe()
        replay_error, replay_location = compare_detailed(
            original_continuation, restored_continuation, "uninterrupted_vs_restored_replay"
        )
        print(
            f"uninterrupted_vs_restored_replay: max_abs_error={replay_error:.3e} "
            f"at {replay_location}"
        )

        caller_restore = receive_ok(
            connection, {"op": "restore", "snapshot_id": snapshot_s1_id}
        )
        caller_state = caller_restore["snapshot_state"]
        caller_errors = report_state_errors(
            "S1_vs_caller_state", state_s1, caller_state
        )
        caller_physics_error = caller_errors["mujoco_integration_state"][0]

        branch = receive_ok(
            connection,
            {
                "op": "branch_rollout",
                "snapshot_id": snapshot_s0_id,
                "candidates": [
                    {"candidate_id": "A-first", "actions": ACTION_A.tolist()},
                    {"candidate_id": "B", "actions": ACTION_B.tolist()},
                    {"candidate_id": "A-repeat", "actions": ACTION_A.tolist()},
                ],
            },
        )
        if branch.get("main_state_restored") is not True:
            raise RuntimeError(
                "Bridge did not confirm main-state restoration: "
                f"{branch.get('main_restore_error', branch.get('main_state_restored'))}"
            )
        candidates = branch.get("candidates", [])
        if len(candidates) != 3:
            raise RuntimeError(f"Expected 3 branch results, received {len(candidates)}")
        for candidate in candidates:
            print_candidate(candidate, verbose_contacts=args.verbose_contacts)
        if any(candidate.get("status") != "ok" for candidate in candidates):
            raise RuntimeError("One or more branch candidates failed")
        candidate_start_details = []
        for candidate in candidates:
            component_errors = report_state_errors(
                f"candidate_{candidate['candidate_id']}_start_vs_S0",
                state_s0,
                candidate["start_state"],
            )
            candidate_start_details.append(max_state_detail(component_errors))
        repeat_error = report_result_error(
            "repeated_candidate_A",
            {
                "start_state": candidates[0]["start_state"],
                "steps": candidates[0]["steps"],
                "executed_steps": candidates[0]["executed_steps"],
            },
            {
                "start_state": candidates[2]["start_state"],
                "steps": candidates[2]["steps"],
                "executed_steps": candidates[2]["executed_steps"],
            },
        )
        after_branch = receive_ok(connection, {"op": "snapshot"})
        main_state_errors = report_state_errors(
            "main_state_after_branch_vs_S1",
            state_s1,
            after_branch["snapshot_state"],
        )
        main_state_error, main_state_location = max_state_detail(main_state_errors)
        branch_preserve_errors = report_state_errors(
            "main_state_after_branch_vs_caller_state",
            caller_state,
            after_branch["snapshot_state"],
        )
        branch_preserve_error, branch_preserve_location = max_state_detail(
            branch_preserve_errors
        )
        max_start_error, max_start_location = max(
            candidate_start_details, key=lambda item: item[0], default=(0.0, "state")
        )
        passed = (
            s0_s1_delta > args.atol
            and probe_delta > args.atol
            and restored_physics_error <= args.atol
            and caller_physics_error <= args.atol
            and replay_error <= args.replay_atol
            and max_start_error <= args.atol
            and repeat_error <= args.atol
            and branch_preserve_error <= args.atol
        )
        print(f"snapshot_S0_id: {snapshot_s0_id}")
        print(f"snapshot_S1_id: {snapshot_s1_id}")
        print(f"main_advance_steps: {main_steps}")
        print(
            f"S0_vs_S1_max_abs_delta: {s0_s1_delta:.3e} "
            f"at {s0_s1_location}"
        )
        print(
            f"S1_restore_probe_state_delta: {probe_delta:.3e} at {probe_location}"
        )
        print(
            f"direct_restore_max_abs_error: {direct_restore_error:.3e} "
            f"at {direct_restore_location}"
        )
        print(
            f"candidate_start_max_abs_error: {max_start_error:.3e} "
            f"at {max_start_location}"
        )
        print(f"repeated_candidate_max_abs_error: {repeat_error:.3e}")
        print(
            "main_state_after_branch_vs_S1_max_abs_error: "
            f"{main_state_error:.3e} at {main_state_location}"
        )
        print(
            "branch_preservation_vs_caller_state_max_abs_error: "
            f"{branch_preserve_error:.3e} at {branch_preserve_location}"
        )
        print(f"physics_and_branch_tolerance: {args.atol:.3e}")
        print(f"continuation_replay_tolerance: {args.replay_atol:.3e}")
        print("S1 derived-state discrepancies above are diagnostics, not pass conditions.")
        print("PASS (bounded replay)" if passed else "FAIL")
        return 0 if passed else 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
