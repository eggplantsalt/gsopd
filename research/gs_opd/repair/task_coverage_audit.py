# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Inventory native goal operators in installed BDDL, without executing tasks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


LOGICAL_OPERATORS = {":goal", "and", "or", "not", "forall", "exists", "imply", "when"}
IMPLEMENTED_OPERATORS = {"on", "in", "open", "close"}


def goal_operators(text: str) -> tuple[list[str], list[str]] | None:
    """Parse the goal expression, treating nested predicate arguments as data."""
    text = re.sub(r";[^\n]*", "", text)
    match = re.search(r"\(:goal\b", text, re.IGNORECASE)
    if match is None:
        return None
    depth = 0
    end = None
    for i in range(match.start(), len(text)):
        depth += (text[i] == "(") - (text[i] == ")")
        if depth == 0:
            end = i + 1
            break
    if end is None:
        raise ValueError("unbalanced_goal_expression")
    tokens = re.findall(r"\(|\)|[^\s()]+", text[match.start():end])
    root: list[Any] = []
    stack: list[list[Any]] = []
    for token in tokens:
        if token == "(":
            node: list[Any] = []
            (stack[-1] if stack else root).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("unexpected_goal_closing_parenthesis")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("goal_token_outside_expression")
    if stack or len(root) != 1:
        raise ValueError("invalid_goal_expression")
    operators: list[str] = []
    logical: list[str] = []

    def visit(node: Any) -> None:
        if not isinstance(node, list) or not node or not isinstance(node[0], str):
            raise ValueError("invalid_goal_node")
        name = node[0].lower()
        if name in LOGICAL_OPERATORS:
            logical.append(name)
            children = node[2:] if name in {"forall", "exists"} else node[1:]
            for child in children:
                # Quantifier declarations are not executable goal predicates.
                if isinstance(child, list) and child and isinstance(child[0], str):
                    visit(child)
        else:
            operators.append(name)

    visit(root[0])
    return operators, logical


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    files = sorted(root.rglob("*.bddl"))
    if not files:
        raise ValueError("no_bddl_files_found")
    suites: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "files": 0, "goal_operator_occurrences": Counter(),
        "logical_operator_occurrences": Counter(), "files_without_goal": [], "parse_errors": [],
    })
    for path in files:
        relative = path.relative_to(root)
        suite = suites[relative.parts[0]]
        suite["files"] += 1
        try:
            parsed = goal_operators(path.read_text(encoding="utf-8"))
            if parsed is None:
                suite["files_without_goal"].append(str(relative))
                continue
            operators, logical = parsed
            suite["goal_operator_occurrences"].update(operators)
            suite["logical_operator_occurrences"].update(logical)
        except (OSError, UnicodeError, ValueError) as exc:
            suite["parse_errors"].append({"path": str(relative), "error": str(exc)})
    all_ops: Counter[str] = Counter()
    for data in suites.values():
        all_ops.update(data["goal_operator_occurrences"])
        data["goal_operator_occurrences"] = dict(sorted(data["goal_operator_occurrences"].items()))
        data["logical_operator_occurrences"] = dict(sorted(data["logical_operator_occurrences"].items()))
    return {
        "schema_version": "gs_opd_bddl_goal_coverage.v1",
        "bddl_root": str(root), "files": len(files),
        "goal_operator_occurrences": dict(sorted(all_ops.items())),
        "implemented_operator_names_not_physical_success_claims": sorted(IMPLEMENTED_OPERATORS),
        "unimplemented_operator_names": sorted(set(all_ops) - IMPLEMENTED_OPERATORS),
        "suites": dict(sorted(suites.items())),
        "scope_note": "Counts are file and predicate occurrences, including generated variants; not a success-rate estimate.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bddl_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": result["files"],
                      "goal_operator_occurrences": result["goal_operator_occurrences"]}))


if __name__ == "__main__":
    main()
