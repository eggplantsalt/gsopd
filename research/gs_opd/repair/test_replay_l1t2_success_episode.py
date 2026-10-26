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
"""Regression checks for the deterministic L1T2 replay source manifest."""

import numpy as np

from research.gs_opd.repair.replay_l1t2_success_episode import (
    _integer_ranges,
    _load_actions,
    _pot2_hand_contact,
    compare_boundary_state,
)


def test_replay_actions_cover_contiguous_original_and_stable_release_sources():
    actions, sources = _load_actions()

    assert len(actions) == 1412
    assert [action.timestep for action in actions] == list(range(1, 1413))
    assert [source["action_count"] for source in sources] == [
        182,
        360,
        56,
        259,
        360,
        118,
        77,
    ]
    assert all(len(action.action) == 7 for action in actions)
    assert all(source["sha256"] for source in sources)
    assert all(source["is_original_historical_action_source"] for source in sources[:6])
    assert not sources[-1]["is_original_historical_action_source"]
    assert sources[-1]["timestep_range"] == [1336, 1412]
    assert all(action.phase == "placement_settle" for action in actions[-20:])


def test_boundary_comparison_distinguishes_bitwise_match_from_roundoff_tolerance():
    expected = np.asarray([0.1, -0.2, 1.0], dtype=np.float64)
    close = expected.copy()
    close[1] += 1e-10
    far = expected.copy()
    far[1] += 1e-4

    close_result = compare_boundary_state(close, expected)
    far_result = compare_boundary_state(far, expected)

    assert close_result["bitwise_exact"] is False
    assert close_result["within_tolerance"] is True
    assert far_result["within_tolerance"] is False


def test_contact_parser_accepts_wrapped_geom_contact_records():
    contact = {
        "geom1": {"name": "gripper0_finger1_pad_collision"},
        "geom2": {"name": "moka_pot_2_g12"},
    }
    assert _pot2_hand_contact({"status": "contact", "contacts": [contact]})
    assert not _pot2_hand_contact({"status": "no_contact", "contacts": []})


def test_contact_timestep_ranges_are_compact_and_lossless():
    assert _integer_ranges([1, 2, 3, 7, 9, 10]) == [[1, 3], [7, 7], [9, 10]]
    assert _integer_ranges([]) == []
