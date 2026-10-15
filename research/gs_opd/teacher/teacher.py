"""High-level Teacher orchestration independent of any VLM provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .backend import RGBFrame, TeacherBackend
from .schema import parse_teacher_output


class SafetyTeacher:
    """Run a configured VLM and accept only schema-valid JSON output."""

    def __init__(self, backend: TeacherBackend):
        self.backend = backend

    def analyze(
        self, instruction: str, images: Sequence[RGBFrame]
    ) -> dict[str, Any]:
        raw_output = self.backend.generate(instruction, images)
        return parse_teacher_output(raw_output)
