"""Contract checks for async launch correlation and conservative attribution."""

import base64
import json
import unittest

from collect_gemm_shapes import control_matches
from summarize_gemm_shapes import summarize


def cpu(name, start, duration, category="cpu_op", args=None, tid=1):
    return {"ph": "X", "name": name, "ts": start, "dur": duration,
            "cat": category, "pid": 10, "tid": tid, "args": args or {}}


def step():
    return cpu("GEMM_DIAG_STEP " + json.dumps({"label": "4k", "phase": "decode",
                                               "active_tokens": 60}), 0, 100)


def encoded_marker(prefix, value):
    payload = json.dumps(value, separators=(",", ":")).encode()
    return prefix + base64.urlsafe_b64encode(payload).decode().rstrip("=")


def kernel(correlation=7, start=1000):
    return {"ph": "X", "name": "mm_kernel_nt", "ts": start, "dur": 5,
            "cat": "kernel", "pid": 0, "tid": 0, "args": {"correlation": correlation}}


class AttributionTests(unittest.TestCase):
    def test_control_trigger_matches_phase_and_active_token_bounds(self):
        control = {"phase": "decode", "min_active_tokens": 48,
                   "max_active_tokens": 64}
        self.assertTrue(control_matches(
            control, {"phase": "decode", "active_tokens": 64}))
        self.assertFalse(control_matches(
            control, {"phase": "mixed", "active_tokens": 64}))
        self.assertFalse(control_matches(
            control, {"phase": "decode", "active_tokens": 32}))

    def test_control_trigger_allows_unbounded_phase(self):
        self.assertTrue(control_matches(
            {"phase": "mixed"}, {"phase": "mixed", "active_tokens": 2048}))

    def test_base64_markers_keep_trace_json_safe(self):
        marker = encoded_marker("GEMM_DIAG_STEP_B64 ", {
            "label": "pilot", "phase": "decode", "active_tokens": 7,
        })
        events = [cpu(marker, 0, 100),
                  cpu("aten::mm", 10, 50,
                      args={"Input Dims": [[8, 4], [4, 3]]}),
                  cpu("mcLaunch", 20, 1, "cuda_runtime", {"correlation": 7}),
                  kernel()]
        result = summarize(json.loads(json.dumps({"traceEvents": events})))
        self.assertEqual(result["rows"][0]["scene"], "pilot")
        self.assertEqual(result["rows"][0]["active_tokens"], 7)

    def test_forward_context_supplies_exact_padded_m_for_compiled_graph(self):
        step_marker = encoded_marker("GEMM_DIAG_STEP_B64 ", {
            "label": "4k", "phase": "decode", "active_tokens": 61,
        })
        forward_marker = encoded_marker("GEMM_DIAG_FORWARD_B64 ", {
            "num_tokens": 64, "graph_tokens": 64, "mode": "FULL",
        })
        events = [cpu(step_marker, 0, 100), cpu(forward_marker, 5, 80),
                  cpu("mcGraphLaunch", 20, 1, "cuda_runtime",
                      {"correlation": 7}), kernel()]
        result = summarize({"traceEvents": events})
        row = result["rows"][0]
        self.assertEqual(row["M"], 64)
        self.assertEqual(row["active_tokens"], 61)
        self.assertEqual(row["M_source"], "forward_context_num_tokens_padded")
        self.assertEqual(result["exact_M_duration_coverage"], 1)

    def test_asynchronous_kernel_uses_launch_id_and_operator_shape(self):
        events = [step(), cpu("aten::mm", 10, 50, args={"Input Dims": [[64, 6144], [6144, 2048]]}),
                  cpu("mcModuleLaunchKernel", 20, 2, "cuda_runtime", {"correlation": 7}), kernel()]
        inventory = [{"module": "model.layers.0.mlp.down_proj", "shape": [2048, 6144]}]
        result = summarize({"traceEvents": events}, inventory)
        self.assertEqual(result["exact_M_duration_coverage"], 1)
        self.assertEqual(result["rows"][0]["M"], 64)
        self.assertEqual(result["rows"][0]["projection_group"], "down_proj")

    def test_graph_padding_is_not_claimed_as_exact_matrix_shape(self):
        graph = cpu("GEMM_DIAG_GRAPH " + json.dumps({"graph_tokens": 64, "action": "replay"}), 10, 50)
        events = [step(), graph, cpu("mcGraphLaunch", 20, 2, "cuda_runtime", {"correlation": 7}), kernel()]
        result = summarize({"traceEvents": events})
        self.assertEqual(result["exact_M_gpu_us"], 0)
        self.assertEqual(result["graph_extent_only_gpu_us"], 5)
        self.assertIsNone(result["rows"][0]["M"])
        self.assertEqual(result["rows"][0]["active_tokens"], 60)

    def test_conflicting_correlation_contexts_are_unknown(self):
        events = [step(), cpu("aten::mm", 10, 20, args={"Input Dims": [[8, 4], [4, 3]]}),
                  cpu("mcLaunch", 15, 1, "cuda_runtime", {"correlation": 7}),
                  cpu("aten::mm", 40, 20, args={"Input Dims": [[16, 4], [4, 3]]}),
                  cpu("mcLaunch", 45, 1, "cuda_runtime", {"correlation": 7}), kernel()]
        result = summarize({"traceEvents": events})
        self.assertEqual(result["unknown_correlation_kernel_calls"], 1)
        self.assertIsNone(result["rows"][0]["M"])

    def test_same_shape_different_projection_names_remain_ambiguous(self):
        events = [step(), cpu("aten::mm", 10, 50, args={"Input Dims": [[8, 2048], [2048, 2048]]}),
                  cpu("mcLaunch", 20, 1, "cuda_runtime", {"correlation": 7}), kernel()]
        inventory = [{"module": "q_proj", "shape": [2048, 2048]},
                     {"module": "o_proj", "shape": [2048, 2048]}]
        result = summarize({"traceEvents": events}, inventory)
        self.assertEqual(result["rows"][0]["projection_group"], "unknown")

    def test_capture_is_excluded_even_when_operator_shapes_exist(self):
        events = [step(), cpu("GEMM_DIAG_GRAPH " + json.dumps({"graph_tokens": 8, "action": "capture"}), 5, 60),
                  cpu("aten::mm", 10, 50, args={"Input Dims": [[8, 4], [4, 3]]}),
                  cpu("mcLaunch", 20, 1, "cuda_runtime", {"correlation": 7}), kernel()]
        result = summarize({"traceEvents": events})
        self.assertEqual(result["capture_excluded_gpu_us"], 5)
        self.assertEqual(result["exact_M_gpu_us"], 0)

    def test_kernel_without_launch_is_not_assigned_by_wall_clock(self):
        result = summarize({"traceEvents": [step(), kernel(start=20)]})
        self.assertEqual(result["unknown_correlation_kernel_calls"], 1)
        self.assertEqual(result["rows"][0]["phase"], "unknown")


if __name__ == "__main__":
    unittest.main()
