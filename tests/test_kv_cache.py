import unittest
from runtime.kv_cache import ModelConfig, KVCacheEstimator, KVCacheMetrics

class TestKVCacheEstimator(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            model_name="Llama-3.2-1B",
            hidden_size=2048,
            num_layers=16,
            num_heads=32,
            head_dim=64,
            dtype_bytes=2.0
        )
        self.estimator = KVCacheEstimator(self.config)

    def test_cache_estimation_calculation(self):
        prompt_tokens = 42
        response_tokens = 103
        metrics = self.estimator.estimate(prompt_tokens, response_tokens)

        # Context length = 42 + 103 = 145
        self.assertEqual(metrics.context_length, 145)
        self.assertEqual(metrics.prompt_tokens, 42)
        self.assertEqual(metrics.response_tokens, 103)

        # Growth per token = 2 * 16 * 2048 * 2 = 131,072 bytes (128 KB)
        expected_growth_bytes = 2 * 16 * 2048 * 2.0
        self.assertEqual(metrics.growth_per_token_bytes, expected_growth_bytes)
        self.assertEqual(metrics.growth_per_token_kb, 128.0)

        # Total cache bytes = 145 * 131072 = 19,005,440 bytes (~18.125 MB)
        expected_cache_bytes = 145 * expected_growth_bytes
        self.assertEqual(metrics.estimated_cache_bytes, expected_cache_bytes)
        self.assertAlmostEqual(metrics.estimated_cache_mb, expected_cache_bytes / (1024 * 1024), places=4)

    def test_explain_behaviour(self):
        metrics = self.estimator.estimate(100, 50)
        explanations = self.estimator.explain_behaviour(metrics)
        self.assertIn("formula", explanations)
        self.assertIn("total_memory", explanations)
        self.assertIn("growth_rate", explanations)
        self.assertIn("multiplier_reason", explanations)

if __name__ == "__main__":
    unittest.main()
