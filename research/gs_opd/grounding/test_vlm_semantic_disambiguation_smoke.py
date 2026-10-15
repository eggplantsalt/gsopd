"""Mocked tests for the VLM semantic disambiguation Smoke."""

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
                    "candidate_count": 1,
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "score": 0.61,
                            "mask_file": str(mask_path.resolve()),
                            "mask_area_pixels": 128,
                        }
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
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return teacher_path, sam_path, iou_path

    def test_resolved_decision_attaches_physical_ids_after_vlm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou = self._fixture(root)
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
            self.assertTrue((out_dir / "semantic_disambiguation_raw.txt").is_file())
            self.assertTrue((out_dir / "semantic_disambiguation.json").is_file())

            payload = backend.calls[0]
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertIn("chat_template_kwargs", payload)
            content = payload["messages"][1]["content"]
            prompt_text = content[0]["text"]
            self.assertIn("yellow mug", prompt_text)
            self.assertIn("Candidate indices shown in the image: [0]", prompt_text)
            self.assertNotIn("0.61", prompt_text)
            self.assertNotIn("0.72", prompt_text)
            self.assertNotIn("physical_body_name", prompt_text)
            self.assertEqual(content[1]["type"], "image_url")

    def test_unresolved_decision_does_not_attach_physical_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, sam, iou = self._fixture(root)
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
