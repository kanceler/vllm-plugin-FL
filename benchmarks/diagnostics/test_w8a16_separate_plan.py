# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for experiment validity; no Torch or Triton required."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from benchmark_w8a16_separate import Launches, compiled_info, load_conversion
from w8a16_separate_plan import (
    CONFIGS, MAIN_MS, PROTOCOL, SEEDS, SHAPES, classify, full_call_gate,
    plan, shortlist, summarize, validation_gate,
)


def fake_kernel(block=1024, warps=4, stages=1, index="BLOCK"):
    return SimpleNamespace(metadata=SimpleNamespace(num_warps=warps,
        num_stages=stages, shared=0), src=SimpleNamespace(constants={index: block}),
        asm={"ttir": "test IR", "binary": b"test binary"}, hash="test-hash")


class ExperimentTests(unittest.TestCase):
    def test_matrix_and_stage_budget(self):
        convert, scale = plan(), plan("scale")
        self.assertEqual(len(CONFIGS), 12)
        self.assertEqual(convert["search_count"], 48)
        self.assertEqual(scale["search_count"], 96)
        self.assertEqual(len(convert["main_points"]), 12)
        self.assertEqual(len(convert["validation_points"]), 14)
        self.assertEqual(len(set(convert["main_points"] + convert["validation_points"])), 26)
        self.assertTrue(set(convert["main_points"]).isdisjoint(convert["validation_points"]))

    def test_exact_one_percent_is_variation(self):
        self.assertEqual(classify(99, 100), "normal_variation")
        self.assertEqual(classify(101, 100), "normal_variation")
        self.assertEqual(classify(98.999, 100), "effective_improvement")
        self.assertEqual(classify(101.001, 100), "effective_regression")

    def test_incomplete_nonfinite_timings_rejected(self):
        for samples in ([1] * 4, [1, 1, 1, 1, float("nan")], [0] * 5):
            with self.assertRaises(ValueError):
                summarize(samples)
        self.assertEqual(summarize([2, 1, 3, 2, 2])["median_ms"], 2)

    def test_shortlist_uses_all_seeds_and_rejects_failures(self):
        records = [
            {"config": [512, 2], "status": "valid", "ratios_to_direct_default": [0.5, 1.2]},
            {"config": [1024, 4], "status": "valid", "ratios_to_direct_default": [0.9, 0.9]},
            {"config": [2048, 4], "status": "failed", "ratios_to_direct_default": [0.1]},
        ]
        self.assertEqual(shortlist(records), [[1024, 4], [512, 2]])

    def test_compiled_requested_actual_mismatch_fails(self):
        with self.assertRaises(RuntimeError):
            compiled_info(fake_kernel(warps=2), 1024, 8, 1)
        with self.assertRaises(RuntimeError):
            compiled_info(fake_kernel(block=512), 1024, 4, 1)
        with self.assertRaises(RuntimeError):
            compiled_info(fake_kernel(stages=2), 1024, 4, 1)

    def test_block_key_supports_conversion_and_scale_versions(self):
        for key, index in (("BLOCK", 3), (3, 3), ((3,), 3), (5, 5), ((5,), 5)):
            self.assertEqual(compiled_info(fake_kernel(index=key), 1024, 4, 1,
                                          block_index=index)["actual"]["block"], 1024)

    def test_launch_uses_jit_for_candidates_and_wrapper_for_control(self):
        calls, kernels = [], {}
        def direct(*args, **kwargs):
            calls.append(kwargs)
            key = (kwargs["BLOCK"], kwargs["num_warps"], kwargs["num_stages"])
            return kernels.setdefault(key, fake_kernel(*key))
        legacy = fake_kernel()
        entry = SimpleNamespace(jit_function=SimpleNamespace(run=direct),
                                run=lambda *a, **k: (legacy, {}))
        backend = SimpleNamespace(weight_int8pack_convert=entry)
        with tempfile.TemporaryDirectory() as path:
            launcher = Launches(None, SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b),
                                backend, Path(path))
            launcher.defaults["convert"] = {"actual": {"warps": 4, "stages": 1}}
            source = SimpleNamespace(numel=lambda: 513)
            objects = [launcher.run("convert", source, None, config=(1024, w))
                       for w in (2, 8, 2)]
            self.assertIs(objects[0], objects[2])
            self.assertIsNot(objects[0], objects[1])
            self.assertEqual([x["num_warps"] for x in calls], [2, 8, 2])
            self.assertIs(launcher.run("convert", source, None), legacy)

    def main_rows(self):
        return [{"m": m, "seed": seed, "role": "main", "bf16_seed_stable": True,
                 "timings": {name: summarize([value]*5) for name, value in
                             (("candidate", 90), ("direct_default", 100),
                              ("production", 100), ("bf16_original", 80))}}
                for m in MAIN_MS for seed in SEEDS]

    def test_full_gate_requires_all_points_and_matched_control(self):
        rows = self.main_rows()
        self.assertTrue(full_call_gate(rows, "candidate"))
        self.assertFalse(full_call_gate(rows[:-1], "candidate"))
        # An entry-only speedup against production must not pass as tile gain.
        rows[0]["timings"]["direct_default"] = summarize([89]*5)
        self.assertFalse(full_call_gate(rows, "candidate"))

    def test_instability_and_validation_regression_veto(self):
        rows = self.main_rows()
        rows[0]["bf16_seed_stable"] = False
        self.assertFalse(full_call_gate(rows, "candidate"))
        row = self.main_rows()[0]
        row.update(role="validation", graph_passed=True)
        self.assertTrue(validation_gate([row], "candidate"))
        row["timings"]["candidate"] = summarize([102]*5)
        self.assertFalse(validation_gate([row], "candidate"))

    def test_scale_rejects_missing_or_expired_conversion_evidence(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "summary.json").write_text(json.dumps({"stage": "convert", "complete": True}))
            (root / "metadata.json").write_text(json.dumps({"fingerprint": {}}))
            with self.assertRaises(ValueError):
                load_conversion(root, {})
            summary = {"stage": "convert", "protocol_complete": True,
                       "execution_complete": True, "eligible_shapes": ["down"],
                       "shapes": {shape: {"accepted_convert_config": [1024, 4],
                           "benefit_verified": True, "validation_complete": True}
                           for shape in SHAPES}}
            (root / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(len(load_conversion(root, {})), 4)
            with self.assertRaises(ValueError):
                load_conversion(root, {"source_hashes": {"bf16_provider": "new"}})
            summary["shapes"]["down"]["validation_complete"] = False
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_conversion(root, {})

    def test_plan_cli_needs_only_standard_library_and_never_overwrites(self):
        script = Path(__file__).with_name("benchmark_w8a16_separate.py")
        with tempfile.TemporaryDirectory() as path:
            target = Path(path) / "plan"
            cmd = [sys.executable, "-S", str(script), "--plan-only", "--output-dir", str(target)]
            first = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            status = json.loads((target / "status.json").read_text())
            self.assertFalse(status["gpu_executed"])
            self.assertFalse(status["complete"])
            second = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)


if __name__ == "__main__":
    unittest.main()
