# SPDX-License-Identifier: Apache-2.0
"""CPU checks of search coverage, pruning, pooling and evidence handling."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from w8a16_splitk_search import (
    CONTROLS, REFINED_MS, SHAPES, WIDE_MS, classify_latency, generate_candidates,
    legality, pool_from_wide, shortlist,
)
from tune_metax_w8a16_splitk import aggregate, append, read_rows, write_json


class SearchTests(unittest.TestCase):
    def candidates(self, m=16, n=2048, k=2048, budget=256 * 1024**2):
        return generate_candidates(m, n, k, sm_count=64, warp_size=64,
                                   workspace_budget=budget)

    def test_exact_approved_matrix(self):
        self.assertEqual(sum(map(len, WIDE_MS.values())), 20)
        self.assertEqual(sum(map(len, REFINED_MS.values())), 40)
        for shape in SHAPES:
            self.assertFalse(set(WIDE_MS[shape]) & set(REFINED_MS[shape]))
            self.assertEqual(len(set(WIDE_MS[shape])), len(WIDE_MS[shape]))
            self.assertEqual(len(set(REFINED_MS[shape])), len(REFINED_MS[shape]))
            self.assertLess(max((*WIDE_MS[shape], *REFINED_MS[shape])), 160)

    def test_legal_bk_s_sets_and_full_domain_accounting(self):
        expected = {64: {2, 4, 8, 16, 32}, 128: {2, 4, 8, 16}, 256: {2, 4, 8}}
        for k in (2048, 6144):
            rows = self.candidates(k=k)
            self.assertEqual(len(rows), 1215)
            legal = [r for r in rows if r["status"] != "pruned_illegal"]
            self.assertEqual(len(legal), 972)
            for bk, splits in expected.items():
                self.assertEqual({r["config"][5] for r in legal if r["config"][2] == bk}, splits)
            self.assertTrue(all(r["config"][5] != 1 for r in rows))

    def test_tiny_m_keeps_all_historical_controls_and_audits(self):
        rows = self.candidates(m=1, k=6144)
        scheduled = {tuple(r["config"]) for r in rows if r["status"] == "scheduled"}
        self.assertTrue(set(CONTROLS) <= scheduled)
        for bm in (16, 32, 64):
            self.assertIn((bm, 64, 64, 4, 2, 32), scheduled)
        self.assertTrue(any(r["status"] == "pruned_heuristic" for r in rows))
        audit = [r for r in rows if "heuristic_audit" in r["sources"]]
        self.assertTrue(audit)
        self.assertTrue(all(r["status"] == "scheduled" for r in audit))

    def test_features_match_split_coverage(self):
        rows = self.candidates(m=16, k=6144)
        for row in rows:
            if "features" not in row:
                continue
            bm, bn, bk, _, _, s = row["config"]
            f = row["features"]
            self.assertEqual(f["k_iters_per_split"] * bk * s, 6144)
            self.assertEqual(f["partial_programs"], ((16 + bm - 1) // bm) * (2048 // bn) * s)
            self.assertEqual(f["workspace_bytes"], 4 * s * 16 * 2048)

    def test_workspace_budget_is_explicit_resource_pruning(self):
        rows = self.candidates(m=64, budget=1)
        self.assertFalse(any(r["status"] == "scheduled" for r in rows))
        self.assertIn("pruned_resource_budget", Counter(r["status"] for r in rows))

    def test_no_n_k_tail_silently_accepted(self):
        self.assertEqual(legality(129, 2048, (16, 64, 128, 4, 2, 8)),
                         "N_not_divisible_by_BN")
        self.assertEqual(legality(2048, 2049, (16, 64, 128, 4, 2, 8)),
                         "K_not_divisible_by_BK_times_S")

    def test_shortlist_ignores_failures_and_preserves_near_structure(self):
        rows = [{"config": [16, 64, 128, 4, 2, 8], "median_ms": 1.0, "status": "passed"},
                {"config": [16, 128, 64, 4, 2, 16], "median_ms": 1.005, "status": "passed"},
                {"config": [16, 32, 64, 4, 2, 32], "median_ms": 0.5, "status": "correctness_failed"},
                {"config": [32, 32, 64, 4, 2, 32], "median_ms": 1.02, "status": "passed"}]
        selected = shortlist(rows, top_k=1)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["median_ms"], 1.0)
        pool = pool_from_wide([{"m": 16, "screening": rows}])
        self.assertTrue(set(CONTROLS) <= {tuple(r["config"]) for r in pool})
        self.assertFalse(any(r["config"] == rows[2]["config"] for r in pool))

    def test_one_percent_boundaries_are_fluctuation(self):
        self.assertEqual(classify_latency(99, 100), "normal_fluctuation")
        self.assertEqual(classify_latency(101, 100), "normal_fluctuation")
        self.assertEqual(classify_latency(98.999, 100), "effective_improvement")
        self.assertEqual(classify_latency(101.001, 100), "effective_regression")

    def test_rejected_finalist_is_not_transferred_as_a_winner(self):
        config = [16, 128, 64, 4, 2, 16]
        point = {"m": 16, "screening": [{"config": config, "status": "passed",
                                         "median_ms": 1.0}],
                 "independent_rejections": [{"config": config}]}
        pool = pool_from_wide([point])
        self.assertFalse(any(r["config"] == config for r in pool))

    def test_incomplete_batch_is_never_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "o" / "m1"
            folder.mkdir(parents=True)
            config = [16, 64, 128, 4, 2, 8]
            write_json(folder / "result.json", {
                "stage": "wide", "selected_config": config,
                "beats_bf16_over_1pct_on_both_seeds": False,
                "results": [{"config": config, "seed": 1, "median_ms": 1.0,
                             "classification": "normal_fluctuation"}]})
            aggregate(root)
            result = json.loads((root / "summary.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["completed_points"], 1)
            self.assertEqual(result["expected_points"], 60)
            self.assertFalse(result["winners"][0]["beats_bf16_over_1pct_on_both_seeds"])

    def test_all_files_with_missing_winner_are_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for shape in SHAPES:
                for m in (*WIDE_MS[shape], *REFINED_MS[shape]):
                    folder = root / shape / f"m{m}"
                    folder.mkdir(parents=True)
                    write_json(folder / "result.json", {
                        "complete": False, "execution_complete": True,
                        "failure_reason": "no_valid_finalist", "stage": "wide",
                        "selected_config": None, "results": [],
                        "beats_bf16_over_1pct_on_both_seeds": False})
            result = aggregate(root)
            self.assertTrue(result["execution_complete"])
            self.assertFalse(result["complete"])
            self.assertEqual(len(result["failed_points"]), 60)

    def test_journal_keeps_failures_and_refuses_torn_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "screening.jsonl"
            append(path, {"config_id": "a", "status": "compile_or_resource_failed"})
            self.assertEqual(read_rows(path)[0]["status"], "compile_or_resource_failed")
            with path.open("a") as stream:
                stream.write('{"broken":')
            with self.assertRaises(RuntimeError):
                read_rows(path)
            target = Path(temp) / "state.json"
            write_json(target, {"complete": False})
            self.assertFalse(json.loads(target.read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
