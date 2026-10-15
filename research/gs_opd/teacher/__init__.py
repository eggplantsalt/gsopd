"""Structured VLM safety semantics for GS-OPD."""

from .backend import (
    BackendConfigError,
    OpenAICompatibleBackend,
    RGBFrame,
    TeacherBackend,
    TeacherBackendError,
    create_backend_from_env,
)
from .schema import (
    SAFETY_OUTPUT_JSON_SCHEMA,
    SCHEMA_VERSION,
    TeacherOutputError,
    parse_teacher_output,
    validate_teacher_output,
)
from .teacher import SafetyTeacher

__all__ = [
    "BackendConfigError",
    "OpenAICompatibleBackend",
    "RGBFrame",
    "SAFETY_OUTPUT_JSON_SCHEMA",
    "SCHEMA_VERSION",
    "SafetyTeacher",
    "TeacherBackend",
    "TeacherBackendError",
    "TeacherOutputError",
    "create_backend_from_env",
    "parse_teacher_output",
    "validate_teacher_output",
]
