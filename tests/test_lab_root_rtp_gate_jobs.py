import json
import tempfile
import unittest
from pathlib import Path


from experiments.day12_rtp_gate.tools import lab_root_rtp_gate_jobs as jobs


class LabRootRtpGateJobsTest(unittest.TestCase):
    def test_known_multi_layer_candidates_are_fixed(self):
        candidates = jobs.known_multi_layer_candidates()

        self.assertEqual(candidates["reverse_3"], [23, 24, 25])
        self.assertEqual(candidates["bi_6"], [2, 11, 20, 21, 23, 24])
        self.assertEqual(candidates["iterative_proxy_k5"], [1, 21, 22, 24, 25])
        self.assertEqual(len(candidates), 12)

    def test_score_command_uses_root_model_trace_and_output_paths(self):
        command = jobs.build_score_command(
            python_bin="/root/env/bin/python",
            root=Path("/root/hs/paper2_layer_pruning"),
            model_path=Path("/root/hs/paper2_layer_pruning/cache/huggingface/model"),
            candidate_name="single_layer_7",
            layers=[7],
            trace_paths=[
                Path("/root/hs/paper2_layer_pruning/results/day12_rtp_gate/traces/calibration.jsonl"),
                Path("/root/hs/paper2_layer_pruning/results/day12_rtp_gate/traces/holdout.jsonl"),
            ],
            output_dir=Path("/root/hs/paper2_layer_pruning/results/day12_rtp_gate/rtd_scores"),
            top_k=100,
            max_seq_tokens=2048,
        )

        self.assertEqual(command[0], "/root/env/bin/python")
        self.assertIn("--root", command)
        self.assertIn("/root/hs/paper2_layer_pruning", command)
        self.assertIn("--model-path", command)
        self.assertIn("/root/hs/paper2_layer_pruning/cache/huggingface/model", command)
        self.assertIn("--candidate-name", command)
        self.assertIn("single_layer_7", command)
        self.assertIn("--runtime-skip-layers", command)
        self.assertIn("7", command)
        self.assertEqual(command.count("--trace-jsonl"), 2)
        self.assertIn("--max-seq-tokens", command)
        self.assertIn("2048", command)

    def test_risky_k5_selects_highest_calibration_rtd(self):
        with tempfile.TemporaryDirectory() as tmp:
            score_dir = Path(tmp)
            payloads = {
                "low_k5.json": {"status": "done", "candidate_name": "low_k5", "runtime_skip_layers": [1, 2, 3, 4, 5], "by_partition": {"calibration": {"rtd": 2.0}}},
                "high_k5.json": {"status": "done", "candidate_name": "high_k5", "runtime_skip_layers": [6, 7, 8, 9, 10], "by_partition": {"calibration": {"rtd": 7.0}}},
                "not_k3.json": {"status": "done", "candidate_name": "not_k3", "runtime_skip_layers": [1, 2, 3], "by_partition": {"calibration": {"rtd": 99.0}}},
            }
            for name, payload in payloads.items():
                (score_dir / name).write_text(json.dumps(payload), encoding="utf-8")

            risky = jobs.select_risky_k5(score_dir)

        self.assertEqual(risky.name, "risky_k5")
        self.assertEqual(risky.layers, [6, 7, 8, 9, 10])
        self.assertEqual(risky.source_candidate_name, "high_k5")
        self.assertEqual(risky.calibration_rtd, 7.0)


if __name__ == "__main__":
    unittest.main()
