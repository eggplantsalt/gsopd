"""Check that a Safety Bridge snapshot replays a fixed action sequence."""

import argparse
from multiprocessing.connection import Client
import sys

import numpy as np


AUTHKEY = b"gs-opd-local"
ACTION_SEQUENCE = np.asarray(
    [
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, -0.04, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=np.float32,
)


def receive_ok(connection, request):
    connection.send(request)
    result = connection.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


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
    raise TypeError(f"Non-numeric snapshot diagnostic: {type(value).__name__}")


def diagnostic_vectors(state):
    return {
        "mujoco_integration": np.asarray(
            state["mujoco_integration_state"], dtype=np.float64
        ),
        "robot": np.asarray(flatten_numeric(state["robot_states"]), dtype=np.float64),
        "objects": np.asarray(flatten_numeric(state["object_states"]), dtype=np.float64),
    }


def same_structure(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_structure(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            same_structure(a, b) for a, b in zip(left, right)
        )
    return isinstance(left, (bool, int, float, np.number)) and isinstance(
        right, (bool, int, float, np.number)
    )


def compare_states(left, right):
    for key in ("mujoco_integration_state", "robot_states", "object_states"):
        if not same_structure(left[key], right[key]):
            raise ValueError(f"{key} diagnostic structure differs")
    left_vectors = diagnostic_vectors(left)
    right_vectors = diagnostic_vectors(right)
    errors = {}
    for name in left_vectors:
        if left_vectors[name].shape != right_vectors[name].shape:
            raise ValueError(
                f"{name} diagnostic shape differs: {left_vectors[name].shape} vs "
                f"{right_vectors[name].shape}"
            )
        difference = np.abs(left_vectors[name] - right_vectors[name])
        errors[name] = float(np.max(difference)) if difference.size else 0.0
    return errors


def execute_sequence(connection, actions):
    states = []
    for step_index, action in enumerate(actions):
        result = receive_ok(
            connection,
            {
                "op": "step",
                "action": action.tolist(),
                "include_snapshot_state": True,
            },
        )
        if "snapshot_state" not in result:
            raise RuntimeError("Bridge step response omitted snapshot_state diagnostics")
        states.append(result["snapshot_state"])
        if result["done"] and step_index + 1 < len(actions):
            print(
                f"Environment terminated after action {step_index + 1}; "
                "the same executed prefix will be replayed.",
                flush=True,
            )
            break
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Safety Bridge host")
    parser.add_argument("--port", type=int, default=38567, help="Safety Bridge port")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Maximum accepted absolute state error (default: 1e-6)",
    )
    args = parser.parse_args()

    if args.atol < 0:
        parser.error("--atol must be non-negative")

    try:
        connection = Client((args.host, args.port), authkey=AUTHKEY)
    except OSError as exc:
        print(f"FAIL: cannot connect to Safety Bridge at {args.host}:{args.port}: {exc}")
        return 1

    try:
        receive_ok(connection, {"op": "reset"})
        snapshot_result = receive_ok(connection, {"op": "snapshot"})
        snapshot_id = snapshot_result["snapshot_id"]
        baseline_state = snapshot_result["snapshot_state"]

        first_pass = execute_sequence(connection, ACTION_SEQUENCE)
        if not first_pass:
            raise RuntimeError("No actions were executed before termination")

        restore_result = receive_ok(
            connection, {"op": "restore", "snapshot_id": snapshot_id}
        )
        restore_errors = compare_states(baseline_state, restore_result["snapshot_state"])

        # If the environment terminated early, replay exactly the same executed prefix.
        replay_actions = ACTION_SEQUENCE[: len(first_pass)]
        second_pass = execute_sequence(connection, replay_actions)
        if len(second_pass) != len(first_pass):
            raise RuntimeError(
                f"Replay step count differs: {len(first_pass)} vs {len(second_pass)}"
            )

        replay_errors = {"mujoco_integration": 0.0, "robot": 0.0, "objects": 0.0}
        for index, (first_state, second_state) in enumerate(zip(first_pass, second_pass)):
            step_errors = compare_states(first_state, second_state)
            replay_errors = {
                name: max(replay_errors[name], error)
                for name, error in step_errors.items()
            }
            print(
                "action {} max_abs_error: {}".format(
                    index + 1,
                    ", ".join(f"{name}={error:.3e}" for name, error in step_errors.items()),
                ),
                flush=True,
            )

        all_errors = list(restore_errors.values()) + list(replay_errors.values())
        passed = all(error <= args.atol for error in all_errors)
        print(f"snapshot_id: {snapshot_id}")
        print(f"actions_replayed: {len(first_pass)}")
        print(
            "restore max_abs_error: "
            + ", ".join(f"{name}={error:.3e}" for name, error in restore_errors.items())
        )
        print(
            "sequence max_abs_error: "
            + ", ".join(f"{name}={error:.3e}" for name, error in replay_errors.items())
        )
        print(f"tolerance: {args.atol:.3e}")
        print("PASS" if passed else "FAIL")
        return 0 if passed else 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
