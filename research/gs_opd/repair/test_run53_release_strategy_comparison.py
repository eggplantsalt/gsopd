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
import numpy as np

from research.gs_opd.repair.run53_release_strategy_comparison import (
    BRIDGE_SOURCE,
    instrument_bridge_source,
    region_inward_shift_xy,
    release_retreat_targets,
    stable_release_candidate,
)


def test_ephemeral_bridge_capture_samples_state_predicates_and_contacts_per_substep():
    original = BRIDGE_SOURCE.read_text(encoding="utf-8")
    modified = instrument_bridge_source(original)

    assert modified != original
    assert '"snapshot_state": _snapshot_diagnostics(canonicalize_kinematics=True)' in modified
    assert '"native_task_status": _native_task_goal_status()' in modified
    assert '"contacts": contacts' in modified
    assert 'result["physical_state_substeps"]' in modified


def test_ephemeral_bridge_capture_does_not_edit_the_research_bridge_file():
    before = BRIDGE_SOURCE.read_bytes()
    instrument_bridge_source(before.decode("utf-8"))
    after = BRIDGE_SOURCE.read_bytes()

    assert after == before


def test_region_inward_shift_is_derived_from_live_geometry_and_bounded():
    shift = region_inward_shift_xy(
        [-0.0486, -0.2096, 0.905],
        [-0.1086, -0.2769, 1.0013],
        0.035,
    )

    assert abs(float((shift**2).sum()) ** 0.5 - 0.035) < 1e-10
    assert shift[0] > 0 and shift[1] > 0


def test_lateral_retreat_moves_away_from_released_object_before_rising():
    eef = [-0.08, -0.30, 1.05]
    pot = [-0.10, -0.32, 1.00]

    lateral, upward = release_retreat_targets(eef, pot, "lateral_up")
    away = lateral[:2] - np.asarray(eef[:2])
    object_to_eef = np.asarray(eef[:2]) - np.asarray(pot[:2])

    assert abs(float((away**2).sum()) ** 0.5 - 0.04) < 1e-10
    assert float((away * object_to_eef).sum()) > 0
    assert lateral[2] == eef[2]
    assert upward[2] == eef[2] + 0.10


def test_retreat_waypoints_reject_unbounded_distances_and_unknown_modes():
    import pytest

    with pytest.raises(ValueError, match="unsupported_release_retreat_style"):
        release_retreat_targets([0, 0, 0], [0.1, 0, 0], "diagonal")
    with pytest.raises(ValueError, match="lateral_retreat_distance"):
        release_retreat_targets([0, 0, 0], [0.1, 0, 0], "lateral_up", 0.1)


def test_stable_release_requires_native_success_contact_free_settling_and_complete_retreat():
    candidate = {
        "status": "complete",
        "substep_state_capture_complete": True,
        "final_native_task_success": True,
        "final_native_predicates": {"pot1_on": True, "pot2_on": True, "stove_on": True},
        "final_hand_pot2_contact": False,
        "all_native_predicates_true_at_every_settle_substep": True,
        "all_settle_substeps_hand_pot2_contact_free": True,
        "pot2_passes_existing_settle_limits": True,
        "withdrawal_displacement_m": 0.10,
        "placement_adjustment": {"retreat_target_remaining_m": 0.006},
    }

    assert stable_release_candidate(candidate)
    assert not stable_release_candidate({**candidate, "all_settle_substeps_hand_pot2_contact_free": False})
    assert not stable_release_candidate({**candidate, "final_native_task_success": False})
    assert not stable_release_candidate({**candidate, "placement_adjustment": {"retreat_target_remaining_m": 0.02}})

