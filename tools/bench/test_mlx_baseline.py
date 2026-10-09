"""Baseline timing checks without importing MLX or requiring a model/GPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import mlx_baseline


class TimingTests(unittest.TestCase):
    def test_prefill_evaluates_only_prefix_cache_and_last_token_logits(self):
        for prompt in ([1], [1, 2, 3]):
            with self.subTest(prompt=prompt):
                inputs, decoded = MagicMock(), MagicMock()
                prefix, last = object(), object()
                inputs.__getitem__.side_effect = [prefix, last] if len(prompt) > 1 else [last]
                mx = MagicMock(uint32="uint32")
                mx.array.side_effect = [inputs, decoded]
                mx.get_peak_memory.return_value = 123
                model = MagicMock()
                model.return_value.__getitem__.return_value = "logits"
                cache = [SimpleNamespace(state="kv", keys=SimpleNamespace(dtype="bfloat16", shape=(1, 2, 256, 8)), nbytes=8192, offset=len(prompt) + 1)]
                with patch.object(mlx_baseline, "footprint_peak", return_value=None):
                    sample, _ = mlx_baseline.measure(model, mx, lambda _: cache, prompt, [7, 8])
                expected_calls = [call(prefix, cache=cache)] if len(prompt) > 1 else []
                expected_calls += [call(last, cache=cache), call(decoded, cache=cache)]
                self.assertEqual(model.call_args_list, expected_calls)
                expected_evals = [call(["kv"])] if len(prompt) > 1 else []
                expected_evals += [call("logits", ["kv"])] * 2
                self.assertEqual(mx.eval.call_args_list, expected_evals)
                self.assertEqual(mx.clear_cache.call_count, int(len(prompt) > 1))
                self.assertEqual(sample["emitted_tokens"], 2)

    def test_decode_excludes_the_first_distribution_and_counts_intervals(self):
        samples = [
            {"token_event_seconds": [1.0, 1.1, 1.2], "emitted_tokens": 3, "ttft_seconds": 1.0, "total_seconds": 1.2, "prefill_seconds": 0.5},
            {"token_event_seconds": [2.0, 2.2, 2.4], "emitted_tokens": 3, "ttft_seconds": 2.0, "total_seconds": 2.4, "prefill_seconds": 1.0},
        ]
        summary = mlx_baseline.summarize(samples, 512)
        self.assertAlmostEqual(summary["decode_tokens_per_second_median"], 7.5)
        self.assertEqual(summary["ttft_median_seconds"], 1.5)
        self.assertEqual(summary["prefill_tokens_per_second_median"], 768)

    def test_one_token_has_no_decode_rate(self):
        summary = mlx_baseline.summarize([{"token_event_seconds": [0.1], "emitted_tokens": 1, "ttft_seconds": 0.1, "total_seconds": 0.1, "prefill_seconds": 0.1}], 1)
        self.assertIsNone(summary["decode_tokens_per_second_median"])

    def test_fully_cached_has_no_prefill_throughput(self):
        sample = {"token_event_seconds": [0.1], "emitted_tokens": 1, "ttft_seconds": 0.1, "total_seconds": 0.1, "prefill_seconds": 0.001, "prefill_tokens": 0}
        self.assertIsNone(mlx_baseline.summarize([sample], 512)["prefill_tokens_per_second_median"])

    def test_invalid_event_records_are_rejected(self):
        for sample in [
            {"token_event_seconds": [1, 0], "emitted_tokens": 2},
            {"token_event_seconds": [1], "emitted_tokens": 2},
        ]:
            with self.assertRaises(ValueError):
                mlx_baseline.summarize([sample], 512)


if __name__ == "__main__":
    unittest.main()
