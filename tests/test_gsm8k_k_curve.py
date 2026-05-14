import unittest

from experiments.day12_rtp_gate.tools.run_gsm8k_k_curve import default_candidates


class Gsm8kKCurveTest(unittest.TestCase):
    def test_default_candidates_cover_k2_and_k5_without_pure_structure_duplicates(self):
        candidates = default_candidates()
        names = [candidate.name for candidate in candidates]

        self.assertEqual(
            names,
            [
                "rtp_gate_k2",
                "rtp_gate_k5",
                "reverse_2",
                "reverse_5",
                "iterative_proxy_k2",
                "iterative_proxy_k5",
            ],
        )
        self.assertNotIn("rtp_gate_pure_k2", names)
        self.assertNotIn("rtp_gate_structure_k2", names)
        self.assertNotIn("rtp_gate_pure_k5", names)
        self.assertNotIn("rtp_gate_structure_k5", names)

    def test_rtp_gate_curve_layers_match_selection_snapshot(self):
        candidates = {candidate.name: candidate.layers for candidate in default_candidates()}

        self.assertEqual(candidates["rtp_gate_k2"], [1, 24])
        self.assertEqual(candidates["rtp_gate_k5"], [1, 9, 10, 19, 24])


if __name__ == "__main__":
    unittest.main()
