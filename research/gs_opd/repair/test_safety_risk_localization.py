import copy
import gzip
import hashlib
import json

import pytest

from research.gs_opd.repair.safety_risk_localization import localize_risks
from research.gs_opd.repair.trajectory_generator_v3 import (
    _localize_cold_start_risks, _screen_current_episode_reconstruction,
)


def _policy():
    return {
        "source": "BDDL",
        "forbidden_objects": [{"object_id": "obstacle", "minimum_clearance_m": 0.010}],
        "allowed_target_objects": [{"object_id": "target", "minimum_clearance_m": 0.010}],
        "allowed_contact_pairs": [{"robot_geom_id": 1, "scene_geom_id": 3, "minimum_clearance_m": -0.002}],
        "forbidden_object_pairs": [{"object_id_a": "target", "object_id_b": "obstacle",
                                    "geom_ids_a": [3], "geom_ids_b": [4]}],
    }


def _record(timestep, *, pair_measurements=True):
    row = {
        "status": "complete",
        "clearances_by_object": {
            "obstacle": {"status": "complete", "minimum_distance_m": 0.02},
            "target": {"status": "complete", "minimum_distance_m": 0.02},
        },
        "allowed_contact_clearances_m": {"1:3": -0.001},
        "self_contacts": [],
        "strict_joint_limit_violations": [],
        "unresolved_contacts": [],
    }
    if pair_measurements:
        row["forbidden_object_pair_clearances"] = {
            "measurement_complete": True,
            "pairs": [{"object_id_a": "target", "object_id_b": "obstacle",
                       "minimum_required_m": 0.01, "measurement": {"minimum_distance_m": 0.02},
                       "verdict": "pass"}],
        }
    return {"timestep": timestep, "phase": "approach", "contact_monitoring": {
        "status": "complete", "captured_physics_substeps": 25,
        "geometry_monitoring": {"substeps": [copy.deepcopy(row) for _ in range(25)]},
        "substep_contacts": [{"physics_substep_index": i, "status": "no_contact", "contacts": []}
                             for i in range(25)],
    }}


def test_first_distance_violation_is_localized_but_missing_pair_distances_are_not_safe():
    record = _record(1, pair_measurements=False)
    record["contact_monitoring"]["geometry_monitoring"]["substeps"][10]["clearances_by_object"]["obstacle"]["minimum_distance_m"] = 0.009
    result = localize_risks([record], _policy(), snapshots=[{"timestep": 0, "path": "t0.pkl"}])
    assert result["first_observed_risk"]["start"]["physics_substep_index"] == 10
    assert result["first_observed_risk"]["nearest_preceding_snapshot"]["path"] == "t0.pkl"
    assert result["missing_forbidden_object_pair_distance_substeps"] == 25
    assert result["status"] == "incomplete"
    assert result["earliest_strict_risk_proven"] is False
    assert result["strict_safety_pass"] is False


def test_localization_rejects_discontinuous_episode_and_accepts_complete_measurements():
    with pytest.raises(ValueError, match="continuous_t0_episode"):
        localize_risks([_record(2)], _policy(), teacher_attested=True)
    result = localize_risks([_record(1)], _policy(), teacher_attested=True)
    assert result["status"] == "incomplete"
    assert result["coverage_limitations"]
    assert result["incomplete_substeps"] == 0
    assert result["first_observed_risk"] is None
    assert result["strict_safety_pass"] is None


def test_v3_localization_requires_continuous_retained_episode_and_writes_provenance(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy()), encoding="utf-8")
    archive = tmp_path / "records.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(1)) + "\n")
    inner = {"safety_policy_provenance": {"policy_path": str(policy_path)}}
    audit = {"success_episode_timestep_continuous": False, "source_action_record_archives": [str(archive)]}
    assert _localize_cold_start_risks(inner, audit, tmp_path)["status"] == "unavailable"
    audit["success_episode_timestep_continuous"] = True
    result = _localize_cold_start_risks(inner, audit, tmp_path)
    assert result["status"] == "incomplete"
    saved = json.loads((tmp_path / "observed_safety_risk_windows.json").read_text(encoding="utf-8"))
    assert saved["policy_sha256"] == hashlib.sha256(policy_path.read_bytes()).hexdigest()
    assert saved["record_archives"] == [str(archive.resolve())]
    assert saved["teacher_attested"] is False


def test_reconstruction_screen_rejects_grasp_report_outside_current_engine(tmp_path):
    external = tmp_path / "historical_report.json"
    external.write_text("{}", encoding="utf-8")
    result = _screen_current_episode_reconstruction(
        None, {"source_search_reports": [str(external)]}, tmp_path / "v3_run",
    )
    assert result == {"status": "unavailable", "reason": "grasp_report_not_from_current_engine"}


def test_localization_detects_self_contact_and_keeps_unresolved_contact_unknown():
    record = _record(1)
    rows = record["contact_monitoring"]["geometry_monitoring"]["substeps"]
    rows[4]["self_contacts"] = [{"geom1": 1, "geom2": 2}]
    rows[5]["unresolved_contacts"] = [{"geom1": 3, "geom2": 4}]
    result = localize_risks([record], _policy(), teacher_attested=True)
    assert result["first_observed_risk"]["start"]["physics_substep_index"] == 4
    assert result["first_observed_risk"]["rules"] == ["robot_self_contact"]
    assert result["incomplete_substeps"] == 1
    assert result["status"] == "incomplete"
