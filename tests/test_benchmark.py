import os
import tempfile
import unittest

from runtime.benchmark import analyze_benchmarks, parse_float, BenchmarkAnalytics


class TestBenchmarkAnalytics(unittest.TestCase):
    def test_parse_float(self):
        self.assertEqual(parse_float("125.50 tok/s"), 125.50)
        self.assertEqual(parse_float("6.25 MB"), 6.25)
        self.assertEqual(parse_float("0.2361 s"), 0.2361)
        self.assertEqual(parse_float("236.10 ms"), 236.10)
        self.assertEqual(parse_float(""), 0.0)

    def test_analyze_benchmarks_with_temp_csv(self):
        csv_content = (
            "Timestamp, Model, Prompt Tokens, Response Tokens, TTFT, Estimated Prefill, Decode Time, Generation Time, Throughput, Estimated KV Cache Memory, Finish Reason\n"
            "2026-07-28T02:00:52, mlx-community/Llama-3.2-1B, 40, 10, 0.2361, 0.2360, 0.0797, 0.3158, 125.50, 6.25 MB, Maximum Token Limit Reached\n"
            "2026-07-28T02:03:23, mlx-community/Llama-3.2-1B, 100, 50, 0.4500, 0.4490, 0.4000, 0.8500, 125.00, 18.75 MB, Stop Token\n"
        )
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp:
            tmp.write(csv_content)
            tmp_path = tmp.name

        try:
            analytics = analyze_benchmarks(tmp_path)
            self.assertIsNotNone(analytics)
            self.assertEqual(analytics.total_runs, 2)
            self.assertEqual(analytics.avg_prompt_tokens, 70.0)
            self.assertEqual(analytics.avg_response_tokens, 30.0)
            self.assertAlmostEqual(analytics.max_kv_cache_mb, 18.75, places=2)
            self.assertTrue(len(analytics.trends) > 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
