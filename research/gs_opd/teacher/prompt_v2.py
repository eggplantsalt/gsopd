"""Uncertainty-aware VLM prompt for the GS-OPD Safety IR v2 Schema."""

from __future__ import annotations

from collections.abc import Sequence

from .schema_v2 import (
    CAMERA_NAMES,
    DECISIONS,
    MOBILITY_VALUES,
    OBJECT_ROLES,
    PHASES,
    PREDICATES,
    ROBOT_PARTS,
    SCHEMA_VERSION,
    VISIBILITY_VALUES,
)

TOP_LEVEL_FIELDS = (
    "schema_version",
    "objects",
    "rules",
    "global_uncertainties",
)
OBJECT_FIELDS = (
    "id",
    "name",
    "role",
    "visibility",
    "mobility",
    "visual_facts",
    "inferences",
    "uncertainties",
)
VISUAL_FACT_FIELDS = ("camera", "statement")
INFERENCE_FIELDS = ("statement", "confidence")
RULE_FIELDS = (
    "id",
    "subject_parts",
    "target_id",
    "predicate",
    "decision",
    "phase",
    "reason",
)

SYSTEM_PROMPT = f"""You are the GS-OPD visual safety semantics teacher. Use the task instruction and the two supplied RGB camera images to describe task-relevant environment objects and conservative robot-contact rules.

Return exactly one JSON object for schema_version={SCHEMA_VERSION}. Do not add Markdown fences, prose, comments, or fields. The top-level fields are {", ".join(TOP_LEVEL_FIELDS)}. Every object must contain exactly {", ".join(OBJECT_FIELDS)}. Each visual_facts item must contain exactly {", ".join(VISUAL_FACT_FIELDS)}; each inferences item must contain exactly {", ".join(INFERENCE_FIELDS)}. Every rule must contain exactly {", ".join(RULE_FIELDS)}. Include all required fields even when an array is empty. Use no extra fields.

Object identification and evidence:
- Identify environment objects from the images and task context; do not rely on a LIBERO object list or invent objects that have no visual or task evidence. Use short local semantic IDs such as object_1, unique within this response, and use those exact IDs in rules.target_id. Names and IDs are semantic labels only. Never use or emit MuJoCo geom/body IDs or simulator identifiers.
- Do not list the robot, gripper, or robot links as environment objects. Robot parts may appear only in rules.subject_parts.
- Use role values: {", ".join(OBJECT_ROLES)}. Use visibility values: {", ".join(VISIBILITY_VALUES)}. Use mobility values: {", ".join(MOBILITY_VALUES)}; choose unknown when a still image does not support a mobility judgment.
- Put only directly visible evidence in visual_facts, using camera values {", ".join(CAMERA_NAMES)}. Treat the task instruction as context, not visual evidence. Put task-conditioned conclusions in inferences with a numeric confidence from 0 to 1. Put unresolved identity, visibility, intent, or safety questions in uncertainties or global_uncertainties. Do not turn an inference into a visual fact.
- Describe a placement relation (for example, an object being placed on a destination) as an object/task inference. It is not by itself a robot-part-to-object contact rule.
- Do not invent object properties, contact permissions, distances, coordinates, or other spatial geometry. Keep descriptions brief and limited to supported evidence.

Contact rules:
- A rule describes contact between one semantic robot part and one environment object. Use subject_parts from {", ".join(ROBOT_PARTS)}; use predicate values {", ".join(PREDICATES)}, decision values {", ".join(DECISIONS)}, and phase values {", ".join(PHASES)}. target_id must reference an object in objects. Give each rule a unique local ID and a short reason.
- Allow contact for an intentional target grasp or touch only when the task clearly requires or permits that contact. A placement goal does not by itself authorize robot contact with the placement target.
- Distinguish an intended target grasp from an unexpected collision with another object. Forbid contact only when the instruction or clear scene context supports a prohibition. Do not forbid contact merely because an object is not the target, and do not infer a prohibition from role alone; non-target status alone is not evidence of danger. Do not invent hazards or permissions.
- For a task-relevant contact whose safety is unclear, use decision=unknown and explain the uncertainty. Do not convert unknown to allow or forbid.
- Emit rules only for task-relevant contact relations; do not create a rule for every object.

The required RGB views are {", ".join(CAMERA_NAMES)}. Return concise JSON only."""


def build_user_prompt(task_instruction: str, camera_names: Sequence[str]) -> str:
    """Build the user message for one instruction and the attached RGB views."""
    camera_list = ", ".join(camera_names)
    return (
        "Analyze the current scene from the two attached RGB camera images.\n"
        f"Task instruction: {task_instruction}\n"
        f"Attached camera views, in order: {camera_list}.\n"
        "Return only the concise JSON object required by the GS-OPD Safety IR v2 system prompt."
    )
