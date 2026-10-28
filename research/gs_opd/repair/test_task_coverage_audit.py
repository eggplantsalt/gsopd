from research.gs_opd.repair.task_coverage_audit import goal_operators, audit


def test_goal_audit_does_not_treat_nested_contact_part_ids_as_predicates(tmp_path):
    path = tmp_path / "affordance" / "L0" / "task.bddl"
    path.parent.mkdir(parents=True)
    path.write_text("(define (:goal (And (CheckGripperContactPart tool (2 3 4)) (On cup plate))))")
    result = audit(tmp_path)
    assert result["goal_operator_occurrences"] == {"checkgrippercontactpart": 1, "on": 1}
    assert result["unimplemented_operator_names"] == ["checkgrippercontactpart"]


def test_goal_audit_reports_missing_and_unbalanced_goals():
    assert goal_operators("(define (:language do something))") is None
    try:
        goal_operators("(define (:goal (And (On a b))")
    except ValueError as exc:
        assert str(exc) == "unbalanced_goal_expression"
    else:
        raise AssertionError("unbalanced goal was accepted")
