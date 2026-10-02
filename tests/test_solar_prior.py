"""Tests for the train-only solar prior built on FAO-56 extraterrestrial radiation."""

import unittest

import numpy as np
import torch

from dmgsr.numerics import coarse
from dmgsr.solar_prior import SolarPrior, extraterrestrial


class SolarPriorTests(unittest.TestCase):
    def setUp(self):
        self.latitude = np.linspace(20.0, 30.0, 10)
        self.keys = ["2000-01", "2000-02", "2000-06", "2000-12"]

    def test_extraterrestrial_is_31_day_padded_and_nonnegative(self):
        ra = extraterrestrial(self.keys, self.latitude)
        self.assertEqual(tuple(ra.shape), (4, 31, 10, 10))
        self.assertTrue(bool(torch.isfinite(ra).all()))
        self.assertTrue(bool((ra >= 0).all()))
        # Padded slots beyond the month length must stay exactly zero.
        self.assertEqual(float(ra[1, 29:].abs().sum()), 0.0)
        # Winter is darker than summer at these northern latitudes.
        self.assertLess(float(ra[0].mean()), float(ra[2].mean()))

    def test_extraterrestrial_is_zero_during_polar_night(self):
        ra = extraterrestrial(["2000-12"], np.array([89.0]))
        self.assertTrue(bool((ra[0, :31] < 1e-6).all()))

    def test_fields_are_month_conserving_and_bounded_by_target(self):
        weights = torch.ones(10, 10)
        target = torch.full((4, 5, 5), 180.0)
        valid = torch.zeros(4, 31, 1, 1)
        for i, key in enumerate(self.keys):
            days = 31 if key in ("2000-01", "2000-12") else (29 if key == "2000-02" else 30)
            valid[i, :days] = 1.0
        prior = SolarPrior(self.latitude, torch.ones(12, 10, 10) * 0.55)
        base, factor = prior.fields(target, self.keys, weights, valid)
        for i, key in enumerate(self.keys):
            days = int(valid[i].sum())
            monthly = coarse(base[i, :days].sum(0) / days, weights)
            torch.testing.assert_close(monthly, target[i], atol=1e-3, rtol=1e-4)
        self.assertTrue(bool((base >= 0).all()))
        # factor = max(Ra, 20) / 400 acts as a normalisation scale, not a bound.
        self.assertTrue(bool((factor > 0).all()))

    def test_fit_uses_only_the_supplied_training_rows(self):
        keys = [f"2000-{month:02d}" for month in range(1, 13)]
        x = torch.full((12, 31, 10, 10), 200.0)
        valid = torch.ones(12, 31, 1, 1)
        valid[:, 28:] = 0.0
        prior = SolarPrior.fit(self.latitude, x, valid, keys, np.arange(12))
        self.assertEqual(tuple(prior.template.shape), (12, 10, 10))
        self.assertTrue(bool(torch.isfinite(prior.template).all()))
        # A constant surface field is exactly recovered as a constant ratio.
        self.assertTrue(bool((prior.template > 0).all()))
        # January has 31 days and no padding, so its ratio is exactly 200/400.
        self.assertAlmostEqual(float(prior.template[0, 0, 0]), 0.6445, places=3)

    def test_fit_rejects_a_month_with_no_valid_training_day(self):
        keys = [f"2000-{month:02d}" for month in range(1, 13)]
        x = torch.full((12, 31, 10, 10), 200.0)
        valid = torch.ones(12, 31, 1, 1)
        # Drop July entirely from the training rows.
        train = np.array([i for i in range(12) if i != 6])
        valid[6] = 0.0
        with self.assertRaises(ValueError):
            SolarPrior.fit(self.latitude, x, valid, keys, train)


if __name__ == "__main__":
    unittest.main()
