"""Mocked tests for Teacher visual point localization and local candidate mapping."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from research.gs_opd.grounding.teacher_visual_point_smoke import (
    parse_visual_point_output,
    run_smoke,
)


class FakeBackend:
    model = "fake-vlm"
    temperature = 0.0
    max_tokens = 300
    thinking_enabled = False

    def __init__(self, response):
        self.response = response
        self.calls = []

    def _post_payload(self, payload, *, schema_constrained=False):
        self.calls.append((payload, schema_constrained))
        return self.response


class TeacherVisualPointSmokeTest(unittest.TestCase):
    def _fixture(self, root: Path):
        root.mkdir(exist_ok=True)
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = 90
        rgb[:, :, 1] = np.arange(16, dtype=np.uint8)[None, :] * 9
        image_path = root / "forward_rgb.png"
        Image.fromarray(rgb).save(image_path)
        geom_image_path = root / "agentview_rgb_rotated_180.png"
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

        mask0 = np.zeros((16, 16), dtype=np.uint8)
        mask0[:, :9] = 255
        mask0_path = root / "candidate_000_mask.png"
        Image.fromarray(mask0).save(mask0_path)
        mask1 = np.zeros((16, 16), dtype=np.uint8)
        mask1[:, 8:] = 255
        mask1_path = root / "candidate_001_mask.png"
        Image.fromarray(mask1).save(mask1_path)

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
                            "statement": "A yellow cup with a handle is visible.",
                        }
                    ],
                    "inferences": [
                        {"statement": "The cup is the intended grasp target.", "confidence": 0.9}
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
                            "score": 0.91,
                            "mask_file": str(mask0_path.resolve()),
                        },
                        {
                            "candidate_index": 1,
                            "score": 0.83,
                            "mask_file": str(mask1_path.resolve()),
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
                            "geom_ids": [11],
                        }
                    ],
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "sam_score": 0.91,
                            "mask_file": str(mask0_path.resolve()),
                            "status": "matched_by_iou",
                            "best_match": {
                                "physical_object_id": "body_7",
                                "body_id": 7,
                                "geom_ids": [11],
                                "iou": 0.8,
                            },
                        },
                        {
                            "candidate_index": 1,
                            "sam_score": 0.83,
                            "mask_file": str(mask1_path.resolve()),
                            "status": "unresolved",
                            "unresolved_reason": "ambiguous_physical_match",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        semantic_path = root / "semantic_disambiguation.json"
        semantic_path.write_text(
            json.dumps(
                {
                    "semantic_decision": {
                        "object_id": "object_1",
                        "candidate_index": 0,
                        "status": "resolved",
                    },
                    "teacher_object": {"object_id": "object_1"},
                    "physical_association": {
                        "status": "matched_by_iou",
                        "body_id": 7,
                        "geom_ids": [11],
                    },
                }
            ),
            encoding="utf-8",
        )
        return teacher_path, image_path, sam_path, iou_path, semantic_path, rgb

    def test_point_maps_to_unique_candidate_and_matches_semantic_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, image, sam, iou, semantic, rgb = self._fixture(root)
            backend = FakeBackend(
                json.dumps(
                    {"status": "located", "x": 0.25, "y": 0.5, "reason": "The visible yellow cup is here."}
                )
            )
            output = root / "out"
            result = run_smoke(
                teacher,
                "object_1",
                image,
                sam,
                iou,
                output,
                backend,
                semantic_result_path=semantic,
            )

            self.assertEqual(len(backend.calls), 1)
            payload, schema_constrained = backend.calls[0]
            self.assertTrue(schema_constrained)
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(result["status"], "resolved")
            self.assertEqual(result["location"]["pixel_x"], 4)
            self.assertEqual(result["location"]["pixel_y"], 8)
            self.assertEqual(result["candidate_membership"]["candidate_indices"], [0])
            self.assertEqual(result["selected_candidate"]["candidate_index"], 0)
            self.assertEqual(result["selected_candidate"]["body_id"], 7)
            self.assertEqual(result["selected_candidate"]["geom_ids"], [11])
            self.assertTrue(result["semantic_comparison"]["candidate_agreement"])
            self.assertTrue(result["semantic_comparison"]["physical_association_agreement"])
            self.assertTrue((output / "teacher_visual_point.png").is_file())
            self.assertTrue((output / "teacher_visual_point_results.json").is_file())
            self.assertTrue((output / "teacher_visual_point_raw.txt").is_file())

            user_content = payload["messages"][1]["content"]
            user_text = " ".join(item["text"] for item in user_content if item["type"] == "text")
            self.assertIn("yellow mug", user_text)
            for forbidden in ("candidate", "mask", "score", "geom", "body", "object_1"):
                self.assertNotIn(forbidden, user_text.lower())
            self.assertEqual(sum(item["type"] == "image_url" for item in user_content), 1)
            schema = payload["response_format"]["json_schema"]["schema"]
            self.assertEqual(set(schema["properties"]), {"status", "x", "y", "reason"})
            data_url = next(item["image_url"]["url"] for item in user_content if item["type"] == "image_url")
            with Image.open(io.BytesIO(base64.b64decode(data_url.partition(",")[2]))) as image_payload:
                np.testing.assert_array_equal(np.asarray(image_payload.convert("RGB")), rgb)

    def test_overlap_and_physical_unresolved_never_select_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            teacher, image, sam, iou, _, _ = self._fixture(root)
            overlap_backend = FakeBackend(
                json.dumps({"status": "located", "x": 0.5, "y": 0.5, "reason": "Visible object point."})
            )
            overlap = run_smoke(teacher, "object_1", image, sam, iou, root / "overlap", overlap_backend)
            self.assertEqual(overlap["status"], "unresolved")
            self.assertEqual(overlap["candidate_membership"]["candidate_indices"], [0, 1])
            self.assertEqual(overlap["physical_association"]["reason"], "point_inside_multiple_candidate_masks")

            physical_backend = FakeBackend(
                json.dumps({"status": "located", "x": 0.9, "y": 0.5, "reason": "Visible object point."})
            )
            physical = run_smoke(teacher, "object_1", image, sam, iou, root / "physical", physical_backend)
            self.assertEqual(physical["status"], "unresolved")
            self.assertEqual(physical["candidate_membership"]["candidate_indices"], [1])
            self.assertIsNone(physical["selected_candidate"])
            self.assertEqual(physical["physical_association"]["reason"], "ambiguous_physical_match")

    def test_parser_accepts_unresolved_and_rejects_invalid_point_payloads(self):
        unresolved = parse_visual_point_output(
            json.dumps({"status": "unresolved", "x": None, "y": None, "reason": "Object is occluded."})
        )
        self.assertEqual(unresolved["status"], "unresolved")
        for raw in (
            '{"status":"located","x":1.1,"y":0.5,"reason":"bad"}',
            '{"status":"located","x":true,"y":0.5,"reason":"bad"}',
            '{"status":"unresolved","x":0.5,"y":null,"reason":"bad"}',
            '{"status":"located","x":0.5,"y":0.5,"reason":"ok","candidate_index":0}',
            '{"status":"located","x":0.5,"x":0.6,"y":0.5,"reason":"duplicate"}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_visual_point_output(raw)


if __name__ == "__main__":
    unittest.main()
