# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read-only interpretation of robosuite 1.4 PandaGripper actuator state."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PANDA_GRIPPER_CLOSE_THRESHOLD = 0.9
PANDA_GRIPPER_ACTUATOR_MAPPING = (
    "robosuite-1.4 PandaGripper.format_action: positive gripper command accumulates "
    "[-1, +1] (close); negative command accumulates [+1, -1] (open)"
)


def _json_safe_raw_action(value: Any) -> Any:
    """Return the observed action in JSON-safe form without reshaping it."""
    try:
        array = np.asarray(value)
    except Exception:
        return repr(value)

    def safe(item: Any) -> Any:
        if isinstance(item, list):
            return [safe(child) for child in item]
        if isinstance(item, tuple):
            return [safe(child) for child in item]
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        if isinstance(item, (int, np.integer)):
            return int(item)
        if isinstance(item, (float, np.floating)):
            number = float(item)
            return number if math.isfinite(number) else repr(number)
        return repr(item)

    return safe(array.tolist())


def parse_panda_gripper_closure(
    current_action: Any,
    *,
    threshold: float = PANDA_GRIPPER_CLOSE_THRESHOLD,
    symmetry_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Interpret the PandaGripper's initial scalar or formatted actuator pair.

    The robosuite 1.4 PandaGripper starts with a one-element zero vector. Its
    first ``format_action`` call changes ``current_action`` to the opposing
    actuator pair. The pair's verified close polarity is negative/positive;
    the opposite polarity is the opening direction. This reports close-command
    progress, not measured finger aperture or grasp success.
    """
    raw = _json_safe_raw_action(current_action)
    result: dict[str, Any] = {
        "status": "unavailable",
        "raw_actuator_action": raw,
        "representation": None,
        "close_command_fraction": None,
        "closure_threshold": float(threshold),
        "closed": False,
        "state": "unavailable",
        "mapping": PANDA_GRIPPER_ACTUATOR_MAPPING,
        "reason": None,
    }

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        result["reason"] = "invalid_gripper_close_threshold"
        return result
    if (
        isinstance(symmetry_tolerance, bool)
        or not isinstance(symmetry_tolerance, (int, float))
        or not math.isfinite(float(symmetry_tolerance))
        or float(symmetry_tolerance) < 0.0
    ):
        result["reason"] = "invalid_actuator_pair_tolerance"
        return result

    try:
        values = np.asarray(current_action, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        result["reason"] = "gripper_current_action_is_not_numeric"
        return result
    if not np.isfinite(values).all():
        result["reason"] = "gripper_current_action_contains_non_finite_values"
        return result

    if values.shape == (1,):
        result["representation"] = "robosuite_initial_single_dof_state"
        if abs(float(values[0])) > float(symmetry_tolerance):
            result["reason"] = "unexpected_nonzero_single_dof_panda_state"
            return result
        # GripperModel initializes current_action to zeros(self.dof). This is
        # a valid pre-format state, but it contains no close command evidence.
        close_fraction = 0.0
        state = "initialized_neutral"
    elif values.shape == (2,):
        result["representation"] = "robosuite_formatted_dual_actuator_state"
        if np.any(values < -1.0) or np.any(values > 1.0):
            result["reason"] = "gripper_actuator_value_outside_normalized_range"
            return result
        first, second = (float(values[0]), float(values[1]))
        if abs(first + second) > float(symmetry_tolerance):
            result["reason"] = "panda_actuator_pair_is_not_equal_and_opposite"
            return result
        if abs(first) <= float(symmetry_tolerance) and abs(second) <= float(symmetry_tolerance):
            close_fraction = 0.0
            state = "neutral"
        elif first < 0.0 and second > 0.0:
            close_fraction = min(-first, second)
            state = "closed_command" if close_fraction >= float(threshold) else "partially_closing"
        elif first > 0.0 and second < 0.0:
            close_fraction = 0.0
            state = "opening"
        else:
            result["reason"] = "panda_actuator_pair_has_unverified_direction"
            return result
    else:
        result["reason"] = f"invalid_panda_current_action_shape:{values.shape}"
        return result

    result.update(
        {
            "status": "pass",
            "close_command_fraction": float(close_fraction),
            "closed": bool(close_fraction >= float(threshold)),
            "state": state,
            "reason": None
            if close_fraction >= float(threshold)
            else "panda_gripper_close_command_below_threshold",
        }
    )
    return result


def validate_panda_gripper_closure_evidence(
    raw_action: Any,
    reported: Any,
    *,
    threshold: float = PANDA_GRIPPER_CLOSE_THRESHOLD,
) -> tuple[bool, dict[str, Any], str | None]:
    """Reparse branch evidence and ensure its structured report agrees."""
    parsed = parse_panda_gripper_closure(raw_action, threshold=threshold)
    if parsed["status"] != "pass":
        return False, parsed, parsed.get("reason")
    if not isinstance(reported, dict):
        return False, parsed, "parsed_gripper_closure_report_missing"
    reported_fraction = reported.get("close_command_fraction")
    if (
        reported.get("status") != parsed["status"]
        or reported.get("closed") is not parsed["closed"]
        or isinstance(reported_fraction, bool)
        or not isinstance(reported_fraction, (int, float))
        or not math.isfinite(float(reported_fraction))
        or abs(float(reported_fraction) - float(parsed["close_command_fraction"])) > 1e-9
        or reported.get("state") != parsed["state"]
    ):
        return False, parsed, "gripper_closure_report_conflicts_with_raw_actuator_action"
    if not parsed["closed"]:
        return False, parsed, parsed.get("reason")
    return True, parsed, None
