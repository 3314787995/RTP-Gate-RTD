import unittest

from experiments.day12_rtp_gate.tools.collect_dense_traces import select_dense_correct_items


class CollectDenseTracesTest(unittest.TestCase):
    def test_default_selection_keeps_original_dense_correct_order(self):
        items = [{"index": i, "correct": i != 2} for i in range(8)]

        selected = select_dense_correct_items(items, total_needed=4, seed=5678, shuffle_correct_items=False)

        self.assertEqual([item["index"] for item in selected], [0, 1, 3, 4])

    def test_shuffle_selection_is_seeded_and_changes_order(self):
        items = [{"index": i, "correct": True} for i in range(20)]

        first = select_dense_correct_items(items, total_needed=8, seed=5678, shuffle_correct_items=True)
        second = select_dense_correct_items(items, total_needed=8, seed=5678, shuffle_correct_items=True)
        default = select_dense_correct_items(items, total_needed=8, seed=5678, shuffle_correct_items=False)

        self.assertEqual([item["index"] for item in first], [item["index"] for item in second])
        self.assertNotEqual([item["index"] for item in first], [item["index"] for item in default])
        self.assertEqual(len(first), 8)


if __name__ == "__main__":
    unittest.main()
