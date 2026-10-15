"""Structured, uncertainty-aware prompt for the GS-OPD Safety Teacher."""

from __future__ import annotations

from collections.abc import Sequence

from .schema import CAMERA_NAMES, SCHEMA_VERSION

SYSTEM_PROMPT = f"""You are the GS-OPD visual safety semantics teacher for robot manipulation.
Given the task instruction and the supplied camera images, describe only task-relevant objects and conservative contact semantics.

Rules:
- Treat the task instruction as task context, not as visual evidence.
- Put directly observable facts only in visual_facts, each tied to the camera that supports it.
- Put conclusions that combine the instruction and images in inferences, with confidence from 0 to 1.
- Put occlusion, ambiguous identity, uncertain intent, and other unresolved points in uncertainties. Do not turn uncertainty into a fact.
- Use role labels target_object, placement_target, avoid_object, tool, obstacle, other, or unknown.
- Use contact_rule=allowed only when the instruction clearly requires or permits the contact; use forbidden only when task context clearly forbids it; otherwise use unknown. Do not treat every contact as dangerous.
- Name semantic robot parts such as gripper, fingers, end_effector, wrist, arm, base, other, or unknown. Never guess a robot link name from pixels.
- Never output MuJoCo geom/body identifiers, simulator IDs, coordinates, or a visual-to-physics mapping. A later Compiler owns physical binding.
- If an object cannot be identified, use a cautious semantic name such as \"unknown object\" and explain the uncertainty.
- Return one JSON object that exactly matches schema version {SCHEMA_VERSION}; no Markdown fences or surrounding prose.

The input cameras are {", ".join(CAMERA_NAMES)}. The required top-level fields are schema_version, objects, and global_uncertainties. Each object must contain name, description, role, visibility, contact_rule, robot_parts, reason, visual_facts, inferences, and uncertainties."""


def build_user_prompt(task_instruction: str, camera_names: Sequence[str]) -> str:
    camera_list = ", ".join(camera_names)
    return (
        "Analyze the current robot scene using the two attached RGB images.\n"
        f"Task instruction: {task_instruction}\n"
        f"Attached camera views, in order: {camera_list}.\n"
        "Include task-relevant target, placement, tool, obstacle, and avoid objects. "
        "If evidence is insufficient, use unknown and state why. Return only the required JSON object."
    )
