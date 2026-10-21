import unittest

from research.gs_opd.smoke.direct_bspline_osc_smoke import _select_route_candidate


def _bundle():
    directions = ("left", "right", "above")
    return {
        "candidates": [
            {
                "candidate_id": f"trajectory-{index + 1:03d}",
                "direction": directions[index % 3],
                "passes_geometric_screen": True,
            }
            for index in range(18)
        ]
    }


class DirectBsplineCandidateSelectionTests(unittest.TestCase):
    def test_omitted_id_preserves_default_first_left_candidate(self):
        selected = _select_route_candidate(_bundle())

        self.assertEqual(selected["candidate_id"], "trajectory-001")
        self.assertEqual(selected["direction"], "left")

    def test_explicit_nondefault_left_candidate_is_selected(self):
        selected = _select_route_candidate(_bundle(), "trajectory-004")

        self.assertEqual(selected["candidate_id"], "trajectory-004")
        self.assertEqual(selected["direction"], "left")

    def test_explicit_candidate_can_select_another_screened_direction(self):
        selected = _select_route_candidate(_bundle(), "trajectory-005")

        self.assertEqual(selected["candidate_id"], "trajectory-005")
        self.assertEqual(selected["direction"], "right")

    def test_explicit_candidate_must_exist_uniquely_and_pass_geometry_screen(self):
        bundle = _bundle()
        bundle["candidates"][3]["passes_geometric_screen"] = False

        with self.assertRaisesRegex(ValueError, "failed_geometric_screen"):
            _select_route_candidate(bundle, "trajectory-004")
        with self.assertRaisesRegex(ValueError, "missing_or_duplicated"):
            _select_route_candidate(bundle, "trajectory-019")


if __name__ == "__main__":
    unittest.main()
