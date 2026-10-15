"""Replaceable VLM backends for structured GS-OPD Teacher inference.
这个文件的作用是，负责把图片和指令发送给VLM，并且取得回复

"""



from __future__ import annotations

import base64
import binascii
import json
import math
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .schema import CAMERA_NAMES
from .prompt_v2 import SYSTEM_PROMPT as SYSTEM_PROMPT_V2
from .prompt_v2 import build_user_prompt as build_user_prompt_v2
from .schema_v2 import CAMERA_NAMES as CAMERA_NAMES_V2
from .schema_v2 import SAFETY_OUTPUT_V2_JSON_SCHEMA


class BackendConfigError(ValueError):
    """Raised when no usable real Teacher backend is configured."""


class TeacherBackendError(RuntimeError):
    """Raised when a configured Teacher endpoint cannot return model text."""


@dataclass(frozen=True)
class RGBFrame:
    camera_name: str
    width: int
    height: int
    rgb_bytes: bytes

    def __post_init__(self) -> None:
        if self.camera_name not in CAMERA_NAMES:
            raise ValueError(f"Unsupported camera name: {self.camera_name!r}")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("RGB frame width and height must be positive")
        if len(self.rgb_bytes) != self.width * self.height * 3:
            raise ValueError(
                f"{self.camera_name} RGB byte count does not match its dimensions"
            )


class TeacherBackend(Protocol):
    """Backend contract; custom local or hosted VLM adapters can implement it."""

    def generate(self, instruction: str, images: Sequence[RGBFrame]) -> str:
        """Return the model's raw text response without repairing its format."""


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    checksum = binascii.crc32(body) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", checksum)


def _rgb_frame_to_data_url(frame: RGBFrame) -> str:
    """Encode HWC RGB bytes as PNG using only the Python standard library."""
    row_size = frame.width * 3
    scanlines = b"".join(
        b"\x00" + frame.rgb_bytes[row * row_size : (row + 1) * row_size]
        for row in range(frame.height)
    )
    ihdr = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(scanlines))
        + _png_chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatibleBackend:
    """Call an OpenAI-compatible multimodal Chat Completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 90.0,
        max_tokens: int = 1800,
        temperature: float = 0.0,
        json_mode: bool = False,
        thinking_enabled: bool | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise BackendConfigError(
                "GS_OPD_TEACHER_BASE_URL must be an http(s) OpenAI-compatible API URL"
            )
        if not model.strip():
            raise BackendConfigError("GS_OPD_TEACHER_MODEL must not be empty")
        if (
            not math.isfinite(timeout_s)
            or timeout_s <= 0
            or max_tokens <= 0
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise BackendConfigError(
                "timeout must be positive, max_tokens positive, and temperature in [0, 2]"
            )
        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            endpoint_path = path
        elif path.endswith("/v1"):
            endpoint_path = f"{path}/chat/completions"
        else:
            endpoint_path = f"{path}/v1/chat/completions"
        self.endpoint = urllib.parse.urlunparse(
            parsed._replace(path=endpoint_path, params="", query="", fragment="")
        )
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.json_mode = bool(json_mode)
        self.thinking_enabled = thinking_enabled

    def _messages(self, instruction: str, images: Sequence[RGBFrame]) -> list[dict]:
        return self._messages_with_prompt(
            instruction,
            images,
            camera_names=CAMERA_NAMES,
            system_prompt=SYSTEM_PROMPT,
            user_prompt_builder=build_user_prompt,
        )

    def _messages_v2(self, instruction: str, images: Sequence[RGBFrame]) -> list[dict]:
        return self._messages_with_prompt(
            instruction,
            images,
            camera_names=CAMERA_NAMES_V2,
            system_prompt=SYSTEM_PROMPT_V2,
            user_prompt_builder=build_user_prompt_v2,
        )

    def _messages_with_prompt(
        self,
        instruction: str,
        images: Sequence[RGBFrame],
        *,
        camera_names: Sequence[str],
        system_prompt: str,
        user_prompt_builder: Callable[[str, Sequence[str]], str],
    ) -> list[dict]:
        by_name = {image.camera_name: image for image in images}
        missing = [camera for camera in camera_names if camera not in by_name]
        if missing:
            raise TeacherBackendError(f"Teacher input is missing cameras: {missing}")
        if len(by_name) != len(images):
            raise TeacherBackendError("Teacher input contains duplicate camera names")

        content: list[dict] = [
            {
                "type": "text",
                "text": user_prompt_builder(instruction, camera_names),
            }
        ]
        for camera in camera_names:
            content.append({"type": "text", "text": f"Camera: {camera}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _rgb_frame_to_data_url(by_name[camera])},
                }
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _extract_content(response: dict) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise TeacherBackendError("VLM response has no choices")
        if not isinstance(choices[0], dict):
            raise TeacherBackendError("VLM response choices[0] is not an object")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise TeacherBackendError("VLM response is missing choices[0].message")
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type", "text") == "text"
                and isinstance(part.get("text"), str)
            ]
            if text_parts:
                return "".join(text_parts)
        raise TeacherBackendError("VLM response content is empty or not text")

    def _post_payload(
        self, payload: dict, *, schema_constrained: bool = False
    ) -> str:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(1200).decode("utf-8", errors="replace")
            if schema_constrained:
                raise TeacherBackendError(
                    "VLM endpoint rejected the Safety IR v2 JSON Schema request; "
                    "the server may not support JSON Schema mode or one or more "
                    "schema keywords. No unconstrained or json_object fallback "
                    f"was attempted. HTTP {exc.code}: {detail}"
                ) from exc
            raise TeacherBackendError(
                f"VLM endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TeacherBackendError(
                f"Cannot reach configured VLM endpoint: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            response_json = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeacherBackendError("VLM endpoint returned invalid JSON") from exc
        if not isinstance(response_json, dict):
            raise TeacherBackendError("VLM endpoint response must be a JSON object")
        return self._extract_content(response_json)

    def generate(self, instruction: str, images: Sequence[RGBFrame]) -> str:
        if not instruction.strip():
            raise ValueError("Task instruction must not be empty")
        payload = {
            "model": self.model,
            "messages": self._messages(instruction, images),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking_enabled is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.thinking_enabled
            }
        return self._post_payload(payload)

    def generate_v2(self, instruction: str, images: Sequence[RGBFrame]) -> str:
        """Generate using the v2 prompt and a mandatory JSON Schema constraint."""
        if not instruction.strip():
            raise ValueError("Task instruction must not be empty")
        payload = {
            "model": self.model,
            "messages": self._messages_v2(instruction, images),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "gs_opd_safety_teacher_v2",
                    "schema": SAFETY_OUTPUT_V2_JSON_SCHEMA,
                },
            },
        }
        if self.thinking_enabled is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.thinking_enabled
            }
        return self._post_payload(payload, schema_constrained=True)


def _read_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise BackendConfigError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise BackendConfigError(f"{name} must be greater than zero")
    return value


def _read_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BackendConfigError(f"{name} must be an integer") from exc
    if value <= 0:
        raise BackendConfigError(f"{name} must be greater than zero")
    return value


def _read_temperature() -> float:
    raw = os.environ.get("GS_OPD_TEACHER_TEMPERATURE", "0")
    try:
        temperature = float(raw)
    except ValueError as exc:
        raise BackendConfigError("GS_OPD_TEACHER_TEMPERATURE must be a number") from exc
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise BackendConfigError(
            "GS_OPD_TEACHER_TEMPERATURE must be finite and in [0, 2]"
        )
    return temperature


def _read_thinking_enabled() -> bool | None:
    raw = os.environ.get("GS_OPD_TEACHER_ENABLE_THINKING")
    if raw is None:
        return None
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise BackendConfigError(
        "GS_OPD_TEACHER_ENABLE_THINKING must be exactly '0' or '1'"
    )


def create_backend_from_env() -> TeacherBackend:
    """Create the configured real backend; never substitute a fake response."""
    backend_name = os.environ.get("GS_OPD_TEACHER_BACKEND", "").strip().lower()
    if not backend_name:
        raise BackendConfigError(
            "No Teacher backend is configured. Set GS_OPD_TEACHER_BACKEND="
            "openai_compatible, GS_OPD_TEACHER_BASE_URL, and GS_OPD_TEACHER_MODEL."
        )
    if backend_name != "openai_compatible":
        raise BackendConfigError(
            f"Unsupported GS_OPD_TEACHER_BACKEND={backend_name!r}; "
            "supported backend: openai_compatible"
        )

    base_url = os.environ.get("GS_OPD_TEACHER_BASE_URL", "").strip()
    model = os.environ.get("GS_OPD_TEACHER_MODEL", "").strip()
    missing = []
    if not base_url:
        missing.append("GS_OPD_TEACHER_BASE_URL")
    if not model:
        missing.append("GS_OPD_TEACHER_MODEL")
    if missing:
        raise BackendConfigError(
            "Missing Teacher backend configuration: " + ", ".join(missing)
        )

    json_mode_value = os.environ.get("GS_OPD_TEACHER_JSON_MODE", "0").strip().lower()
    if json_mode_value not in ("0", "1", "false", "true", "no", "yes"):
        raise BackendConfigError(
            "GS_OPD_TEACHER_JSON_MODE must be one of 0/1/false/true/no/yes"
        )
    try:
        return OpenAICompatibleBackend(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("GS_OPD_TEACHER_API_KEY") or None,
            timeout_s=_read_positive_float("GS_OPD_TEACHER_TIMEOUT_S", 90.0),
            max_tokens=_read_positive_int("GS_OPD_TEACHER_MAX_TOKENS", 1800),
            temperature=_read_temperature(),
            json_mode=json_mode_value in ("1", "true", "yes"),
            thinking_enabled=_read_thinking_enabled(),
        )
    except ValueError as exc:
        if isinstance(exc, BackendConfigError):
            raise
        raise BackendConfigError(f"Invalid Teacher backend configuration: {exc}") from exc
