import json
import os
import tempfile
import unittest

from replay import find_latest_run_file, replay_session


class TestReplayEngine(unittest.TestCase):
    def test_find_latest_run_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "run_2026_07_28_010000.json")
            file2 = os.path.join(tmp_dir, "run_2026_07_28_020000.json")
            
            with open(file1, "w") as f:
                f.write("{}")
            with open(file2, "w") as f:
                f.write("{}")

            # Force file2 modification time to be newer
            os.utime(file1, (100, 100))
            os.utime(file2, (200, 200))

            latest = find_latest_run_file(tmp_dir)
            self.assertEqual(latest, file2)

    def test_replay_session_runs_without_exceptions(self):
        snapshot_data = {
            "timestamp": "2026-07-28T02:00:00",
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "prompt": "Test prompt",
            "formatted_prompt": "Test prompt formatted",
            "token_ids": [1, 2, 3],
            "decoded_tokens": "Test prompt formatted",
            "prompt_tokens": 3,
            "response": "Test response",
            "response_tokens": 5,
            "max_tokens": 25,
            "tokenization_time": 0.0001,
            "ttft": 0.1,
            "estimated_prefill": 0.09,
            "decode_time": 0.05,
            "generation_time": 0.15,
            "throughput": 100.0,
            "estimated_kv_cache_mb": 5.0,
            "finish_reason": "Stop Token",
            "context_length": 8,
            "num_layers": 16,
            "hidden_size": 2048,
            "dtype_bytes": 2.0,
            "growth_per_token_kb": 128.0,
            "per_layer_mb": 0.31
        }

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
            json.dump(snapshot_data, tmp)
            tmp_path = tmp.name

        try:
            # Replay session should complete cleanly without errors
            replay_session(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
