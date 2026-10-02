"""Tests for the area-weighted grid operators."""

import unittest

import torch

from dmgsr.numerics import coarse, expand, project


class NumericsTests(unittest.TestCase):
    def test_coarse_matches_manual_area_weighted_mean(self):
        weights = torch.arange(1.0, 101.0).reshape(10, 10)
        x = torch.rand(2, 5, 10, 10)
        got = coarse(x, weights)
        self.assertEqual(tuple(got.shape), (2, 5, 5, 5))
        block = x[0, 0, :2, :2]
        manual = (block * weights[:2, :2]).sum() / weights[:2, :2].sum()
        torch.testing.assert_close(got[0, 0, 0, 0], manual)

    def test_expand_is_the_2x_nearest_neighbour_inverse_of_blocking(self):
        x = torch.rand(3, 5, 5)
        y = expand(x)
        self.assertEqual(tuple(y.shape), (3, 10, 10))
        torch.testing.assert_close(y[:, 0::2, 0::2], x)
        torch.testing.assert_close(y[:, 1::2, 1::2], x)

    def test_expand_then_coarse_is_identity_for_uniform_weights(self):
        weights = torch.ones(10, 10)
        x = torch.rand(4, 5, 5)
        torch.testing.assert_close(coarse(expand(x), weights), x)

    def test_project_reproduces_target_coarse_mean(self):
        weights = torch.ones(10, 10)
        mask = torch.ones(2, 31, 1, 1)
        values = torch.rand(2, 31, 10, 10) * 200.0
        target = torch.full((2, 5, 5), 100.0)
        got = project(values, target, weights, mask)
        torch.testing.assert_close(coarse(got.mean(1), weights), target, atol=1e-4, rtol=1e-5)
        self.assertTrue(bool((got >= 0).all()))

    def test_project_marks_padded_days_invalid(self):
        weights = torch.ones(10, 10)
        mask = torch.zeros(1, 31, 1, 1)
        mask[:, :28] = 1.0
        values = torch.rand(1, 31, 10, 10) * 200.0
        got = project(values, torch.full((1, 5, 5), 90.0), weights, mask)
        self.assertEqual(float(got[:, 28:].abs().sum()), 0.0)
        torch.testing.assert_close(coarse(got[:, :28].mean(1), weights), torch.full((1, 5, 5), 90.0), atol=1e-4, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
