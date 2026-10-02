"""Tests for the velocity-diffusion schedule and the monthly projection."""

import unittest

import torch

from dmgsr.diffusion_ops import (
    alphas,
    bounded_partial_project,
    consistent_noise,
    diffusion_chunks,
    partial_project,
    recover,
)
from dmgsr.numerics import coarse


def valid_days(count, batch=1):
    mask = torch.zeros(batch, 31)
    mask[:, :count] = 1.0
    return mask


class ScheduleTests(unittest.TestCase):
    def test_alphas_is_positive_and_decreasing(self):
        a = alphas(torch.device("cpu"))
        self.assertEqual(a.shape, (64,))
        self.assertTrue(bool((a > 0).all()))
        self.assertTrue(bool((a[1:] < a[:-1]).all()))

    def test_velocity_recovery_inverts_the_forward_process(self):
        x = torch.randn(2, 31, 10, 10)
        e = torch.randn_like(x)
        for a in (torch.tensor(1e-6), torch.tensor(0.5), torch.tensor(0.99)):
            z = a.sqrt() * x + (1 - a).sqrt() * e
            v = a.sqrt() * e - (1 - a).sqrt() * x
            torch.testing.assert_close(recover(z, v, a), x, atol=1e-6, rtol=1e-5)

    def test_consistent_noise_reproduces_the_current_latent(self):
        z = torch.randn(2, 31, 10, 10)
        changed = torch.randn_like(z)
        a = torch.tensor(0.8)
        noise = consistent_noise(z, changed, a)
        torch.testing.assert_close(a.sqrt() * changed + (1 - a).sqrt() * noise, z)

    def test_diffusion_chunks_cover_the_reverse_schedule_once(self):
        self.assertEqual(diffusion_chunks(10, 4), [(6, 9), (2, 5), (0, 1)])
        chunks = diffusion_chunks(64, 8)
        flattened = [step for low, high in chunks for step in range(high, low - 1, -1)]
        self.assertEqual(flattened, list(range(63, -1, -1)))

    def test_diffusion_chunks_rejects_nonpositive_arguments(self):
        for steps, size in ((0, 8), (64, 0), (-1, 8)):
            with self.assertRaises(ValueError):
                diffusion_chunks(steps, size)


def block_energy(field):
    """Coarse per-block energy: [batch, 31, 10, 10] -> [batch, 5, 5]."""
    return field.reshape(field.shape[0], 31, 5, 2, 5, 2).sum((1, -1, -3))


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.weights = torch.ones(10, 10)
        self.target = torch.full((1, 5, 5), 100.0)

    def assert_energy_invariant(self, values, target, weights, valid, observed, got, count):
        """Check the documented projection guarantees per coarse block.

        * A block with hidden cells and observations below the monthly
          requirement ``target * valid_days`` is filled up to it.
        * A block whose observed cells already meet or exceed that requirement
          keeps its supplied energy, i.e. its hidden cells are left alone
          instead of being scaled down or blanked out.
        * A block with no hidden cell is left exactly as supplied.
        """
        mask = valid[:, :, None, None]
        supplied = block_energy(values.clamp_min(0.0) * mask)
        observed_energy = block_energy(values.clamp_min(0.0) * observed * mask)
        hidden = block_energy((1.0 - observed) * mask)
        required = target * count * 4
        result = block_energy(got)
        for i in range(5):
            for j in range(5):
                if hidden[0, i, j] <= 0:
                    expected = float(observed_energy[0, i, j])
                elif float(observed_energy[0, i, j]) >= float(required[0, i, j]):
                    expected = float(supplied[0, i, j])
                else:
                    expected = float(required[0, i, j])
                self.assertAlmostEqual(float(result[0, i, j]), expected, delta=0.02 * count)

    def test_projection_preserves_observed_cells_and_padding(self):
        valid = valid_days(31)
        observed = torch.ones(1, 31, 1, 1)
        values = torch.rand(1, 31, 10, 10) * 200.0
        got = partial_project(values, self.target, self.weights, valid, observed)
        torch.testing.assert_close(got, values.clamp_min(0.0))

    def test_projection_changes_only_missing_days(self):
        valid = valid_days(31)
        observed = torch.ones(1, 31, 1, 1)
        observed[:, 10:17] = 0.0
        values = torch.rand(1, 31, 10, 10) * 200.0
        got = partial_project(values, self.target, self.weights, valid, observed)
        torch.testing.assert_close(got[:, observed[0, :, 0, 0] > 0], values[:, observed[0, :, 0, 0] > 0].clamp_min(0.0))

    def test_projection_hits_the_target_energy_in_every_disturbed_block(self):
        """The energy target must use valid days, not the 31-day padded length.

        ``partial_project`` deliberately leaves blocks that contain no gap at
        their supplied value, so the invariant is per coarse block: a block
        with hidden cells must carry ``target * valid_days * 4`` and a block
        without any must keep ``value * valid_days * 4``.
        """
        for count in (31, 30, 28):
            with self.subTest(valid_days=count):
                valid = valid_days(count)
                observed = valid[:, :, None, None].repeat(1, 1, 10, 10)
                observed[:, 5:12, 2:6, 2:6] = 0.0
                values = torch.full((1, 31, 10, 10), 80.0) * valid[:, :, None, None]
                got = partial_project(values, self.target, self.weights, valid, observed)
                self.assert_energy_invariant(values, self.target, self.weights, valid, observed, got, count)
                # Missing cells filled, observations preserved, padding zero.
                torch.testing.assert_close(got[:, valid[0].sum().int():], torch.zeros_like(got[:, valid[0].sum().int():]))
                sel = observed > 0
                torch.testing.assert_close(got[sel], values[sel])

    def test_projection_uses_valid_days_not_padding_for_short_months(self):
        """A 28-day month must not be credited with 31 days of energy.

        Block (2, 1) covers rows 4-5 and columns 2-3, which this gap hides
        completely for 7 days.  Its remaining energy is
        ``count * 100 * 4 - (count - 7) * 80 * 4``; because only 7 days are
        written back, the per-cell fill is ``(300 - 60 * (count - 7) / count)``
        W/m2: 168.57 for 31 days, 165.71 for 30 and 160.0 for 28.  The pre-fix
        code always used 31 days, giving 202.86 for a 28-day month.
        """
        for count, expected_fill in ((31, 168.5714), (30, 165.7143), (28, 160.0)):
            with self.subTest(valid_days=count):
                valid = valid_days(count)
                observed = valid[:, :, None, None].repeat(1, 1, 10, 10)
                observed[:, 5:12, 2:6, 2:6] = 0.0
                values = torch.full((1, 31, 10, 10), 80.0) * valid[:, :, None, None]
                got = partial_project(values, self.target, self.weights, valid, observed)
                # Block (2, 1) spans rows 4-5 and columns 2-3.
                fill = got[0, 5:7, 4:6, 2:4]
                torch.testing.assert_close(fill, torch.full((2, 2, 2), expected_fill), atol=1e-3, rtol=1e-4)

    def test_projection_fills_a_zero_energy_missing_block(self):
        valid = valid_days(31)
        observed = torch.ones(1, 31, 1, 1)
        observed[:, 10:17] = 0.0
        values = torch.ones(1, 31, 10, 10) * 50.0
        values[:, 10:17] = 0.0
        got = partial_project(values, self.target, self.weights, valid, observed)
        torch.testing.assert_close(coarse(got.sum(1) / 31, self.weights), self.target, atol=1e-4, rtol=1e-5)
        self.assertTrue(bool((got[:, 10:17] > 0).all()))

    def test_bounded_projection_respects_the_upper_bound(self):
        valid = valid_days(31)
        observed = torch.ones(1, 31, 10, 10)
        observed[:, 4:7] = 0.0
        values = torch.full((1, 31, 10, 10), 1000.0)
        upper = torch.full_like(values, 300.0)
        got = bounded_partial_project(values, self.target, self.weights, valid, observed, upper)
        missing = observed <= 0
        self.assertTrue(bool((got[missing] <= upper[missing] + 1e-5).all()))
        self.assertTrue(bool(torch.equal(got[~missing], values[~missing])))

    def test_bounded_projection_is_energy_consistent_when_the_bound_is_slack(self):
        valid = valid_days(28)
        observed = valid[:, :, None, None].repeat(1, 1, 10, 10)
        observed[:, 3:9, 1:4, 1:4] = 0.0
        values = torch.full((1, 31, 10, 10), 120.0) * valid[:, :, None, None]
        # A hidden block whose observations already exceed the 100 W/m2
        # requirement keeps its observed energy; hidden cells below it are
        # filled up to the requirement.
        upper = torch.full_like(values, 1e4)
        got = bounded_partial_project(values, self.target, self.weights, valid, observed, upper)
        self.assert_energy_invariant(values, self.target, self.weights, valid, observed, got, 28)
        # The untouched corner block keeps its supplied value.
        self.assertAlmostEqual(float(block_energy(got)[0, 4, 4]), 120.0 * 28 * 4, delta=0.6)
        self.assertEqual(float(got[:, 28:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
