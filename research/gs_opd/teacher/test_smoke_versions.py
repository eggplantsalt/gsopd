"""Offline request and parser-selection tests for Teacher Smoke versions."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

import research.gs_opd.teacher.smoke as smoke_module
from research.gs_opd.teacher.backend import OpenAICompatibleBackend
from research.gs_opd.teacher.prompt import SYSTEM_PROMPT as SYSTEM_PROMPT_V1
from research.gs_opd.teacher.prompt import build_user_prompt as build_user_prompt_v1
from research.gs_opd.teacher.prompt_v2 import SYSTEM_PROMPT as SYSTEM_PROMPT_V2
from research.gs_opd.teacher.prompt_v2 import (
    build_user_prompt as build_user_prompt_v2,
)
from research.gs_opd.teacher.schema import (
    CAMERA_NAMES as CAMERA_NAMES_V1,
    SCHEMA_VERSION as SCHEMA_VERSION_V1,
    parse_teacher_output,
)
from research.gs_opd.teacher.schema_v2 import (
    CAMERA_NAMES as CAMERA_NAMES_V2,
    SAFETY_OUTPUT_V2_JSON_SCHEMA,
    SCHEMA_VERSION as SCHEMA_VERSION_V2,
    parse_teacher_output_v2,
)


TASK_INSTRUCTION = "pick up the red cup"


class _HTTPResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def read(self) -> bytes:
        return self.body


class _BridgeConnection:
    def __init__(self):
        self.request = None
        self.closed = False

    def send(self, request) -> None:
        self.request = request

    def recv(self) -> dict:
        return {
            "instruction": TASK_INSTRUCTION,
            "images": {
                camera_name: {
                    "shape": (1, 1, 3),
                    "dtype": "uint8",
                    "data": bytes((1, 2, 3)),
                }
                for camera_name in CAMERA_NAMES_V1
            },
        }

    def close(self) -> None:
        self.closed = True


class TeacherSmokeVersionTests(unittest.TestCase):
    def _run_smoke(self, version: str) -> tuple[dict, object, object, object]:
        schema_version = SCHEMA_VERSION_V1 if version == "v1" else SCHEMA_VERSION_V2
        raw_output = json.dumps(
            {
                "schema_version": schema_version,
                "objects": [],
                **({"rules": []} if version == "v2" else {}),
                "global_uncertainties": [],
            }
        )
        http_response = _HTTPResponse(
            json.dumps(
                {"choices": [{"message": {"content": raw_output}}]}
            ).encode("utf-8")
        )
        backend = OpenAICompatibleBackend(
            base_url="http://127.0.0.1:8000/v1",
            model="offline-test-model",
            json_mode=True,
            thinking_enabled=False,
        )
        bridge = _BridgeConnection()
        captured_stdout = io.StringIO()
        argv = [] if version == "v1" else ["--teacher-version", version]

        with patch.object(smoke_module, "Client", return_value=bridge):
            with patch.object(
                smoke_module, "create_backend_from_env", return_value=backend
            ):
                with patch(
                    "urllib.request.urlopen", return_value=http_response
                ) as urlopen_mock:
                    with patch.object(
                        smoke_module,
                        "parse_teacher_output",
                        wraps=parse_teacher_output,
                    ) as v1_parser:
                        with patch.object(
                            smoke_module,
                            "parse_teacher_output_v2",
                            wraps=parse_teacher_output_v2,
                        ) as v2_parser:
                            with contextlib.redirect_stdout(captured_stdout):
                                exit_code = smoke_module.main(argv)

        self.assertEqual(exit_code, 0)
        self.assertTrue(bridge.closed)
        urlopen_mock.assert_called_once()
        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        if version == "v1":
            v1_parser.assert_called_once_with(raw_output)
            v2_parser.assert_not_called()
        else:
            v1_parser.assert_not_called()
            v2_parser.assert_called_once_with(raw_output)
        return payload, v1_parser, v2_parser, urlopen_mock

    def test_default_v1_uses_v1_prompt_json_mode_and_parser_once(self) -> None:
        payload, _v1_parser, _v2_parser, _urlopen_mock = self._run_smoke("v1")
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_PROMPT_V1)
        self.assertEqual(
            payload["messages"][1]["content"][0]["text"],
            build_user_prompt_v1(TASK_INSTRUCTION, CAMERA_NAMES_V1),
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_v2_uses_v2_prompt_schema_constraint_and_parser_once(self) -> None:
        payload, _v1_parser, _v2_parser, _urlopen_mock = self._run_smoke("v2")
        self.assertEqual(payload["messages"][0]["content"], SYSTEM_PROMPT_V2)
        self.assertEqual(
            payload["messages"][1]["content"][0]["text"],
            build_user_prompt_v2(TASK_INSTRUCTION, CAMERA_NAMES_V2),
        )
        response_format = payload["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(
            response_format["json_schema"]["name"], "gs_opd_safety_teacher_v2"
        )
        generated_schema = response_format["json_schema"]["schema"]
        expected_schema = copy.deepcopy(SAFETY_OUTPUT_V2_JSON_SCHEMA)
        expected_subject_parts = expected_schema["properties"]["rules"]["items"]
        expected_subject_parts = expected_subject_parts["properties"]["subject_parts"]
        del expected_subject_parts["uniqueItems"]
        self.assertEqual(generated_schema, expected_schema)

        original_subject_parts = SAFETY_OUTPUT_V2_JSON_SCHEMA["properties"]["rules"]
        original_subject_parts = original_subject_parts["items"]["properties"]
        original_subject_parts = original_subject_parts["subject_parts"]
        self.assertTrue(original_subject_parts["uniqueItems"])
        generated_subject_parts = generated_schema["properties"]["rules"]["items"]
        generated_subject_parts = generated_subject_parts["properties"][
            "subject_parts"
        ]
        self.assertNotIn("uniqueItems", generated_subject_parts)
        self.assertEqual(generated_subject_parts["minItems"], 1)
        self.assertEqual(
            generated_subject_parts["items"]["enum"],
            original_subject_parts["items"]["enum"],
        )
        self.assertEqual(
            generated_schema["properties"]["rules"]["items"]["required"],
            SAFETY_OUTPUT_V2_JSON_SCHEMA["properties"]["rules"]["items"][
                "required"
            ],
        )
        self.assertFalse(
            generated_schema["properties"]["rules"]["items"][
                "additionalProperties"
            ]
        )

    def test_v2_schema_rejection_is_explicit_and_has_no_fallback(self) -> None:
        backend = OpenAICompatibleBackend(
            base_url="http://127.0.0.1:8000/v1",
            model="offline-test-model",
        )
        error = urllib.error.HTTPError(
            backend.endpoint,
            400,
            "unsupported schema keyword",
            hdrs=None,
            fp=io.BytesIO(b"uniqueItems is not supported"),
        )
        with patch("urllib.request.urlopen", side_effect=error) as urlopen_mock:
            with self.assertRaisesRegex(
                smoke_module.TeacherBackendError,
                "JSON Schema request.*No unconstrained or json_object fallback",
            ):
                backend.generate_v2(
                    TASK_INSTRUCTION,
                    [
                        smoke_module.RGBFrame(camera, 1, 1, bytes((1, 2, 3)))
                        for camera in CAMERA_NAMES_V1
                    ],
                )
        urlopen_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
