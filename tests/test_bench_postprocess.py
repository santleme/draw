"""Paridade da alternativa medida, inclusive empates entre classes."""

from itertools import product

import numpy as np
import pytest

from bench_postprocess import argmax_mask, benchmark, direct_mask


@pytest.mark.parametrize("output_shape", [(720, 1280), (512, 512)])
def test_direct_comparisons_match_argmax_for_finite_logits_and_ties(output_shape):
    logits = np.array(list(product((-1.0, 0.0, 1.0), repeat=4)), dtype=np.float32)
    logits = logits.T.reshape(1, 4, 9, 9)
    expected = argmax_mask(logits, output_shape)

    np.testing.assert_array_equal(direct_mask(logits, output_shape), expected)
    assert expected.shape == output_shape
    assert expected.dtype == np.uint8
    assert set(np.unique(expected)) == {0, 255}


def test_benchmark_returns_mean_and_p95_without_timing_thresholds():
    logits = np.random.default_rng(0).standard_normal((1, 4, 8, 8)).astype(np.float32)
    results = benchmark(logits, (16, 24), iterations=3, warmup=1)

    assert set(results) == {"argmax", "direct"}
    for stats in results.values():
        assert set(stats) == {"mean_ms", "p95_ms"}
        assert all(np.isfinite(value) and value >= 0 for value in stats.values())
