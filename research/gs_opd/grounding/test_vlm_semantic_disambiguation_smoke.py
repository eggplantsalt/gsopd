"""Mocked tests for the VLM semantic disambiguation Smoke."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from research.gs_opd.grounding.vlm_semantic_disambiguation_smoke import (
    parse_disambiguation_output,
    run_smoke,
)


class FakeBackend:
    model = "fake-vlm"
    temperature = 0.0
    max_tokens = 400
    thinking_enabled = False

    def __init__(self, response):
        self.response = response
        self.calls = []

    def _post_payload(self, payload):
        self.calls.append(payload)
        return self.response


class VlmSemanticDisambiguationSmokeTest(unittest.TestCase):
    def _fixture(self, root: Path):
        root.mkdir(exist_ok=True)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 120
        rgb[:, :, 1] = np.arange(16, dtype=np.uint8)[None, :] * 8
        image_path = root / "forward_rgb.png"
        Image.fromarray(rgb).save(image_path)
        geom_image_path = root / "agentview_512_rgb_rotated_180.png"
        Image.fromarray(rgb).save(geom_image_path)
        geom_metadata_path = root / "geom_metadata.json"
        geom_metadata_path.write_text(
            json.dumps(
                {
                    "width": 16,
                    "height": 16,
                    "rgb_rotated_180_file": geom_image_path.name,
                    "coordinate_directions": {
                        geom_image_path.name: "raw image rotated 180 degrees; SAM 3 forward orientation"
                    },
                }
            ),
            encoding="utf-8",
        )

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:, :8] = 255
        mask_path = root / "candidate_000_mask.png"
        Image.fromarray(mask).save(mask_path)
        robot_mask = np.zeros((16, 16), dtype=np.uint8)
        robot_mask[:, 8:] = 255
        robot_mask_path = root / "candidate_001_mask.png"
        Image.fromarray(robot_mask).save(robot_mask_path)

        teacher = {
            "schema_version": "gs_opd.safety_teacher.v2",
            "objects": [
                {
                    "id": "object_1",
                    "name": "yellow mug",
                    "role": "target_object",
                    "visibility": "visible",
                    "mobility": "likely_movable",
                    "visual_facts": [
                        {
                            "camera": "agentview_image",
                            "statement": "A yellow cylindrical cup with a handle is visible.",
                        }
                    ],
                    "inferences": [
                        {"statement": "The mug is the object intended for grasping.", "confidence": 0.9}
                    ],
                    "uncertainties": [],
                }
            ],
            "rules": [],
            "global_uncertainties": [],
        }
        teacher_path = root / "teacher.json"
        teacher_path.write_text(json.dumps(teacher), encoding="utf-8")

        sam_path = root / "sam3_results.json"
        sam_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "object_id": "object_1",
                    "image": str(image_path.resolve()),
                    "image_width": 16,
                    "image_height": 16,
                    "candidate_count": 2,
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "score": 0.61,
                            "mask_file": str(mask_path.resolve()),
                            "mask_area_pixels": 128,
                        },
                        {
                            "candidate_index": 1,
                            "score": 0.58,
                            "mask_file": str(robot_mask_path.resolve()),
                            "mask_area_pixels": 128,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        iou_path = root / "iou_results.json"
        iou_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "coordinate_direction": "180-degree forward orientation shared with SAM 3 input",
                    "image_width": 16,
                    "image_height": 16,
                    "sam3_results": str(sam_path.resolve()),
                    "geom_metadata": str(geom_metadata_path.resolve()),
                    "physical_objects": [
                        {
                            "physical_object_id": "body_7",
                            "body_id": 7,
                            "body_name": "physical_body_name",
                            "geom_ids": [11, 12],
                            "geom_names": ["geom_a", "geom_b"],
                        }
                    ],
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "sam_score": 0.61,
                            "mask_file": str(mask_path.resolve()),
                            "status": "matched_by_iou",
                            "best_match": {
                                "physical_object_id": "body_7",
                                "body_id": 7,
                                "body_name": "physical_body_name",
                                "geom_ids": [11, 12],
                                "geom_names": ["geom_a", "geom_b"],
                                "iou": 0.72,
                            },
                        },
                        {
                            "candidate_index": 1,
                            "sam_score": 0.58,
                            "mask_file": str(robot_mask_path.resolve()),
                            "status": "matched_by_iou",
                            "best_match": {
                                "physical_object_id": "body_8",
                                "body_id": 8,
                                "body_name": "robot_link_body",
                                "geom_ids": [13],
                                "geom_names": ["robot_geom"],
                                "iou": 0.9,
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        geometry_inventory_path = root / "bridge_geometry_inventory.json"
        geometry_inventory_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "robot_classification_status": "available",
                    "geometries": [
                        {"geom_id": 11, "geom_name": "geom_a", "body_id": 7, "body_name": "physical_body_name", "category": "scene"},
                        {"geom_id": 12, "geom_name": "geom_b", "body_id": 7, "body_name": "physical_body_name", "category": "scene"},
                        {"geom_id": 13, "geom_name": "robot_geom", "body_id": 8, "body_name": "robot_link_body", "category": "robot"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return teacher_path, sam_path, iou_path, geometry_inventory_path

    def test_resolved_decision_attaches_physical_ids_after_vlm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou, _geometry = self._fixture(root)
            backend = FakeBackend(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": 0,
                        "reason": "The numbered region covers the yellow handled cup.",
                        "status": "resolved",
                    }
                )
            )
            out_dir = root / "out"
            result = run_smoke(teacher, "object_1", sam, iou, out_dir, backend)

            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(result["semantic_decision"]["status"], "resolved")
            self.assertEqual(result["physical_association"]["body_id"], 7)
            self.assertEqual(result["physical_association"]["geom_ids"], [11, 12])
            self.assertEqual(result["status"], "resolved")
            self.assertTrue((out_dir / "numbered_candidates.png").is_file())
            self.assertTrue((out_dir / "original_rgb.png").is_file())
            self.assertTrue((out_dir / "semantic_disambiguation_raw.txt").is_file())
            self.assertTrue((out_dir / "semantic_disambiguation.json").is_file())

            with Image.open(teacher.parent / "forward_rgb.png") as image:
                original_rgb = np.asarray(image.convert("RGB"))
            with Image.open(out_dir / "original_rgb.png") as image:
                saved_rgb = np.asarray(image.convert("RGB"))
            np.testing.assert_array_equal(saved_rgb, original_rgb)
            with Image.open(out_dir / "numbered_candidates.png") as image:
                annotated_rgb = np.asarray(image.convert("RGB"))
            # Away from the number glyph, the candidate interior remains the
            # original pixel; only its thin perimeter and label are drawn.
            np.testing.assert_array_equal(annotated_rgb[14, 4], original_rgb[14, 4])
            self.assertFalse(np.array_equal(annotated_rgb[8, 7], original_rgb[8, 7]))

            payload = backend.calls[0]
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertIn("chat_template_kwargs", payload)
            content = payload["messages"][1]["content"]
            prompt_text = content[0]["text"]
            self.assertIn("yellow mug", prompt_text)
            self.assertIn("Candidate indices shown in the image: [0, 1]", prompt_text)
            self.assertIn("completely unmodified original RGB", prompt_text)
            self.assertIn("must not be used to judge color or material", prompt_text)
            self.assertNotIn("0.61", prompt_text)
            self.assertNotIn("0.72", prompt_text)
            self.assertNotIn("physical_body_name", prompt_text)
            self.assertEqual(content[1]["type"], "text")
            self.assertEqual(content[2]["type"], "image_url")
            self.assertEqual(content[3]["type"], "text")
            self.assertEqual(content[4]["type"], "image_url")
            self.assertIn("Image 1", content[1]["text"])
            self.assertIn("Image 2", content[3]["text"])
            image_arrays = []
            for content_index in (2, 4):
                data_url = content[content_index]["image_url"]["url"]
                encoded = data_url.partition(",")[2]
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    image_arrays.append(np.asarray(image.convert("RGB")))
            np.testing.assert_array_equal(image_arrays[0], original_rgb)
            np.testing.assert_array_equal(image_arrays[1], annotated_rgb)

    def test_unresolved_decision_does_not_attach_physical_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou, _geometry = self._fixture(root)
            backend = FakeBackend(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": None,
                        "reason": "The target is partly occluded.",
                        "status": "unresolved",
                    }
                )
            )
            result = run_smoke(teacher, "object_1", sam, iou, root / "out", backend)
            self.assertEqual(result["status"], "unresolved")
            self.assertIsNone(result["selected_candidate"])
            self.assertEqual(result["physical_association"]["status"], "unresolved")
            self.assertEqual(len(backend.calls), 1)

    def test_active_robot_geometry_is_filtered_before_vlm_candidate_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou, geometry = self._fixture(root)
            backend = FakeBackend(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": 0,
                        "reason": "The visual region depicts the described cup.",
                        "status": "resolved",
                    }
                )
            )
            result = run_smoke(
                teacher,
                "object_1",
                sam,
                iou,
                root / "filtered_out",
                backend,
                geometry_inventory_path=geometry,
            )
            content = backend.calls[0]["messages"][1]["content"]
            self.assertIn("Candidate indices shown in the image: [0]", content[0]["text"])
            self.assertEqual(result["robot_candidates_filtered"], [1])
            self.assertEqual(result["physical_association"]["status"], "matched_by_iou")

    def test_task_instruction_is_optional_disambiguation_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou, _geometry = self._fixture(root)
            backend = FakeBackend(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": 0,
                        "reason": "The candidate matches the described object.",
                        "status": "resolved",
                    }
                )
            )
            run_smoke(
                teacher,
                "object_1",
                sam,
                iou,
                root / "instruction_out",
                backend,
                task_instruction="pick up the mug from the stove",
            )
            prompt_text = backend.calls[0]["messages"][1]["content"][0]["text"]
            self.assertIn("Task instruction: pick up the mug from the stove", prompt_text)

    def test_structured_result_rejects_unknown_candidate_and_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "available candidate_index"):
            parse_disambiguation_output(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": 99,
                        "reason": "uncertain",
                        "status": "resolved",
                    }
                ),
                "object_1",
                [0, 1],
            )
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_disambiguation_output(
                json.dumps(
                    {
                        "object_id": "object_1",
                        "candidate_index": None,
                        "reason": "uncertain",
                        "status": "unresolved",
                        "sam_score": 0.99,
                    }
                ),
                "object_1",
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
