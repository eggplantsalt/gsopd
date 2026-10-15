"""Lightweight tests for SAM 3 / MuJoCo segmentation IoU matching."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from research.gs_opd.grounding.sam3_mujoco_iou_smoke import (
    _load_inputs,
    run_matching,
)


class Sam3MujocoIouSmokeTest(unittest.TestCase):
    def _make_fixture(self, root: Path):
        capture_dir = root / "capture"
        sam_dir = root / "sam"
        capture_dir.mkdir()
        sam_dir.mkdir()

        rows, cols = np.indices((8, 8))
        rgb = np.stack((rows * 20, cols * 20, (rows + cols) * 10), axis=-1).astype(
            np.uint8
        )
        geom_ids = np.full((8, 8), -1, dtype=np.int32)
        geom_ids[:4, :4] = 0
        geom_ids[4:, :4] = 1
        geom_ids[:, 4:] = 2
        Image.fromarray(rgb).save(capture_dir / "agentview_512_rgb_rotated_180.png")
        np.save(capture_dir / "geom_ids_rotated_180.npy", geom_ids, allow_pickle=False)

        geom_metadata = {
            "width": 8,
            "height": 8,
            "model_geom_count": 3,
            "rgb_rotated_180_file": "agentview_512_rgb_rotated_180.png",
            "geom_ids_rotated_180_file": "geom_ids_rotated_180.npy",
            "coordinate_directions": {
                "agentview_512_rgb_rotated_180.png": "raw image rotated 180 degrees; SAM 3 forward orientation",
                "geom_ids_rotated_180.npy": "raw label map rotated 180 degrees; aligned with SAM 3 forward orientation",
            },
            "visible_geoms": [
                {
                    "geom_id": 0,
                    "geom_name": "geom_a",
                    "body_id": 10,
                    "body_name": "body_a",
                    "pixel_count": 16,
                },
                {
                    "geom_id": 1,
                    "geom_name": "geom_b",
                    "body_id": 10,
                    "body_name": "body_a",
                    "pixel_count": 16,
                },
                {
                    "geom_id": 2,
                    "geom_name": "geom_c",
                    "body_id": 20,
                    "body_name": "body_c",
                    "pixel_count": 32,
                },
            ],
        }
        geom_metadata_path = capture_dir / "geom_metadata.json"
        geom_metadata_path.write_text(json.dumps(geom_metadata), encoding="utf-8")

        sam_image_path = sam_dir / "input.png"
        Image.fromarray(rgb).save(sam_image_path)
        body_a_mask = np.zeros((8, 8), dtype=np.uint8)
        body_a_mask[:, :4] = 255
        no_overlap_mask = np.zeros((8, 8), dtype=np.uint8)
        # Empty Candidate is unresolved even when SAM assigns it a high score.
        tied_mask = np.zeros((8, 8), dtype=np.uint8)
        tied_mask[0, :] = 255
        Image.fromarray(body_a_mask).save(sam_dir / "candidate_000_mask.png")
        Image.fromarray(no_overlap_mask).save(sam_dir / "candidate_001_mask.png")
        Image.fromarray(tied_mask).save(sam_dir / "candidate_002_mask.png")
        sam_results_path = sam_dir / "sam3_results.json"
        sam_results_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "image": str(sam_image_path.resolve()),
                    "image_width": 8,
                    "image_height": 8,
                    "candidate_count": 3,
                    "candidates": [
                        {
                            "candidate_index": 0,
                            "score": 0.2,
                            "mask_file": "candidate_000_mask.png",
                        },
                        {
                            "candidate_index": 1,
                            "score": 0.99,
                            "mask_file": "candidate_001_mask.png",
                        },
                        {
                            "candidate_index": 2,
                            "score": 0.8,
                            "mask_file": "candidate_002_mask.png",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return (
            sam_results_path,
            capture_dir / "geom_ids_rotated_180.npy",
            geom_metadata_path,
            sam_image_path,
        )

    def test_groups_geoms_by_body_and_scores_all_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sam_results, geom_ids, metadata, _ = self._make_fixture(root)
            result = run_matching(sam_results, geom_ids, metadata, root / "out")

            self.assertEqual(len(result["physical_objects"]), 2)
            self.assertEqual(result["physical_objects"][0]["geom_ids"], [0, 1])
            self.assertEqual(result["iou_matrix"]["candidate_indices"], [0, 1, 2])
            self.assertEqual(result["iou_matrix"]["values"][0], [1.0, 0.0])
            self.assertEqual(result["candidates"][0]["sam_score"], 0.2)
            self.assertEqual(result["candidates"][0]["status"], "matched_by_iou")
            self.assertEqual(result["candidates"][0]["best_overlap_objects"][0]["body_id"], 10)
            self.assertEqual(result["candidates"][1]["sam_score"], 0.99)
            self.assertEqual(result["candidates"][1]["status"], "unresolved")
            self.assertEqual(result["candidates"][1]["unresolved_reason"], "empty_candidate_mask")
            self.assertEqual(result["candidates"][2]["status"], "unresolved")
            self.assertEqual(result["candidates"][2]["unresolved_reason"], "tied_best_iou")
            self.assertEqual(len(result["candidates"][2]["best_overlap_objects"]), 2)
            self.assertEqual(len(result["comparison_images"]), 3)
            for filename in result["comparison_images"]:
                self.assertTrue((root / "out" / filename).is_file())

    def test_rejects_sam_image_not_pixel_identical_to_forward_capture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sam_results, geom_ids, metadata, sam_image = self._make_fixture(root)
            wrong = np.asarray(Image.open(sam_image).convert("RGB")).copy()
            wrong[0, 0, 0] ^= 1
            Image.fromarray(wrong).save(sam_image)
            with self.assertRaisesRegex(RuntimeError, "not pixel-identical"):
                _load_inputs(sam_results, geom_ids, metadata)

    def test_rejects_wrong_coordinate_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sam_results, geom_ids, metadata, _ = self._make_fixture(root)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["coordinate_directions"]["geom_ids_rotated_180.npy"] = "raw renderer order"
            metadata.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not marked as the 180-degree"):
                _load_inputs(sam_results, geom_ids, metadata)


if __name__ == "__main__":
    unittest.main()
