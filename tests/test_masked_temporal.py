"""Tests for the masked weather-conditioned temporal diffusion model."""

import unittest

import numpy as np
import torch

from dmgsr.masked_temporal import (
    DATA_ROOT,
    MaskedTemporalVelocityNet,
    graph_boundary_projection,
    graph_laplacian,
    graph_structure_loss,
    monthly_conditions,
    random_observation_mask,
    spatial_boundary_blend,
    spatial_diffusion_blend,
    summarize_masked_metrics,
)


class MaskTests(unittest.TestCase):
    def test_random_mask_preserves_valid_shape_and_hides_values(self):
        valid = torch.ones(4, 31, 1, 1)
        mask = random_observation_mask(valid, np.random.default_rng(3), min_gap=3, max_gap=5, spatial_probability=0.0)
        self.assertEqual(mask.shape, valid.shape)
        self.assertTrue(bool(((mask == 0) | (mask == 1)).all()))
        self.assertTrue(bool((mask.sum((1, 2, 3)) < 31).all()))

    def test_every_gap_mode_stays_inside_the_valid_mask(self):
        valid = torch.ones(2, 31, 10, 10)
        for mode in ("temporal", "spatial", "mixed", "endpoint"):
            mask = random_observation_mask(valid, np.random.default_rng(4), 2, 4, mode=mode)
            self.assertTrue(bool((mask <= valid).all()), mode)
            self.assertTrue(bool((mask.sum() < valid.sum())), mode)

    def test_spatial_mode_never_removes_a_whole_month(self):
        valid = torch.ones(3, 31, 10, 10)
        valid[:, 30:] = 0.0
        mask = random_observation_mask(valid, np.random.default_rng(5), 7, 7, mode="spatial")
        self.assertTrue(bool((mask.sum((1, 2, 3)) > 0).all()))

    def test_endpoint_gap_touches_a_month_boundary(self):
        valid = torch.ones(8, 31, 1, 1)
        mask = random_observation_mask(valid, np.random.default_rng(6), 3, 3, mode="endpoint")
        touched = [(float(mask[i, 0, 0, 0]) == 0) or (float(mask[i, 30, 0, 0]) == 0) for i in range(mask.shape[0])]
        self.assertTrue(all(touched))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            random_observation_mask(torch.ones(1, 31, 1, 1), np.random.default_rng(1), mode="diagonal")


class NetworkTests(unittest.TestCase):
    def test_network_accepts_explicit_mask_and_weather_channels(self):
        torch.manual_seed(2)
        net = MaskedTemporalVelocityNet(width=4).eval()
        z = torch.randn(2, 31, 10, 10)
        monthly = torch.randn(2, 4, 10, 10)
        weather = torch.randn(2, 31, 3, 10, 10)
        observed_values = torch.randn(2, 31, 10, 10)
        observed = torch.ones(2, 31, 10, 10)
        valid = torch.ones(2, 31, 10, 10)
        output = net(z, monthly, weather, observed_values, observed, torch.ones(2) * 0.5, valid)
        self.assertEqual(output.shape, z.shape)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_network_output_is_zero_on_invalid_days(self):
        torch.manual_seed(3)
        net = MaskedTemporalVelocityNet(width=4).eval()
        valid = torch.ones(1, 31, 10, 10)
        valid[:, 30:] = 0.0
        with torch.no_grad():
            output = net(
                torch.randn(1, 31, 10, 10), torch.randn(1, 4, 10, 10), torch.randn(1, 31, 3, 10, 10),
                torch.randn(1, 31, 10, 10), torch.ones(1, 31, 10, 10), torch.zeros(1), valid,
            )
        self.assertEqual(float(output[:, 30:].abs().sum()), 0.0)

    def test_monthly_conditions_are_scale_normalised(self):
        keys = ["2000-01", "2000-07"]
        target = torch.full((2, 5, 5), 200.0)
        base = torch.full((2, 31, 10, 10), 150.0)
        valid = torch.ones(2, 31, 10, 10)
        valid[:, 28:] = 0.0
        conditions = monthly_conditions(target, keys, mean_scale=250.0, base=base, valid=valid)
        self.assertEqual(tuple(conditions.shape), (2, 4, 10, 10))
        torch.testing.assert_close(conditions[:, 0], (target / 250.0).repeat_interleave(2, -2).repeat_interleave(2, -1))
        # January and July must carry opposite seasonal phase.
        self.assertAlmostEqual(float(conditions[0, 1, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(conditions[0, 2, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(conditions[1, 1, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(conditions[1, 2, 0, 0]), -1.0, places=5)
        # The monthly baseline channel is the valid-day mean of the solar base.
        torch.testing.assert_close(conditions[:, 3], torch.full((2, 10, 10), 150.0 / 250.0))


class GraphTests(unittest.TestCase):
    def test_laplacian_is_zero_on_a_constant_field(self):
        lap = graph_laplacian(torch.full((2, 3, 6, 6), 7.0))
        torch.testing.assert_close(lap, torch.zeros_like(lap), atol=1e-5, rtol=0)

    def test_laplacian_detects_a_local_bump(self):
        field = torch.zeros(1, 1, 5, 5)
        field[0, 0, 2, 2] = 1.0
        lap = graph_laplacian(field)
        self.assertGreater(float(lap[0, 0, 2, 2]), 0.0)
        self.assertAlmostEqual(float(lap[0, 0, 0, 0]), 0.0, places=6)

    def test_structure_loss_is_zero_for_identical_fields(self):
        truth = torch.rand(1, 2, 6, 6)
        loss = graph_structure_loss(truth.clone(), truth)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_boundary_projection_only_touches_cells_next_to_observations(self):
        valid = torch.ones(1, 1, 5, 5)
        observed = torch.ones(1, 1, 5, 5)
        observed[0, 0, 2, 2] = 0.0
        # Varying values so that the neighbour average differs from the centre.
        values = torch.arange(25.0).reshape(1, 1, 5, 5)
        centre = float(values[0, 0, 2, 2])
        got = graph_boundary_projection(values, observed, valid, strength=0.15)
        neighbours = (float(values[0, 0, 1, 2]) + float(values[0, 0, 3, 2])
                      + float(values[0, 0, 2, 1]) + float(values[0, 0, 2, 3])) / 4.0
        self.assertAlmostEqual(float(got[0, 0, 2, 2]), centre + 0.15 * (neighbours - centre), places=4)
        # A cell with no observed neighbour and the observed cells stay put.
        self.assertAlmostEqual(float(got[0, 0, 0, 0]), float(values[0, 0, 0, 0]), places=5)
        self.assertAlmostEqual(float(got[0, 0, 4, 4]), float(values[0, 0, 4, 4]), places=5)
        self.assertAlmostEqual(float(got[0, 0, 1, 2]), float(values[0, 0, 1, 2]), places=5)

    def test_boundary_projection_with_zero_strength_is_identity(self):
        valid = torch.ones(1, 1, 5, 5)
        observed = torch.ones(1, 1, 5, 5)
        observed[0, 0, 2, 2] = 0.0
        values = torch.rand(1, 1, 5, 5)
        torch.testing.assert_close(graph_boundary_projection(values, observed, valid, strength=0.0), values)

    def test_blends_never_modify_observed_cells(self):
        valid = torch.ones(2, 3, 6, 6)
        observed = torch.ones(2, 3, 6, 6)
        observed[:, :, 1:3, 1:4] = 0.0
        values = torch.rand(2, 3, 6, 6) * 100.0
        for blend in (spatial_boundary_blend, spatial_diffusion_blend):
            got = blend(values, observed, valid, blend=0.35)
            sel = observed > 0
            torch.testing.assert_close(got[sel], values[sel])


class MetricTests(unittest.TestCase):
    def test_summary_scores_only_missing_cells(self):
        torch.manual_seed(7)
        valid = torch.ones(1, 31, 10, 10)
        observed = torch.ones(1, 31, 10, 10)
        observed[:, 5:8, 2:4, 2:4] = 0.0
        truth = torch.full((1, 31, 10, 10), 100.0)
        samples = truth[:, None].repeat(1, 6, 1, 1, 1)
        samples[:, 0] = 90.0
        samples[:, -1] = 110.0
        summary = summarize_masked_metrics(samples, truth, observed, valid)
        missing = (valid > 0) & (observed <= 0)
        self.assertAlmostEqual(summary["pooled_missing_fraction"], float(missing.float().mean()), places=6)
        # The ensemble mean equals the truth on missing cells, so RMSE is zero.
        self.assertAlmostEqual(summary["pooled_rmse"], 0.0, places=5)
        self.assertGreater(summary["pooled_crps"], 0.0)

    def test_summary_rejects_an_all_observed_mask(self):
        valid = torch.ones(1, 31, 4, 4)
        samples = torch.ones(1, 3, 31, 4, 4)
        with self.assertRaises(ValueError):
            summarize_masked_metrics(samples, valid, valid.clone(), valid)


class DataTests(unittest.TestCase):
    def test_bundled_power_pilot_has_the_expected_grid_and_period(self):
        if not DATA_ROOT.exists():
            self.skipTest("bundled NASA POWER feature bundle is absent")
        from dmgsr.daily_features import FEATURES, open_daily_feature_bundle

        bundle = open_daily_feature_bundle(DATA_ROOT)
        self.assertEqual(bundle.gsr.sizes["lat"], 10)
        self.assertEqual(bundle.gsr.sizes["lon"], 10)
        self.assertEqual(bundle.gsr.sizes["time"], 15 * 365 + 4)
        self.assertEqual(int(bundle.features.sizes["feature"]), len(FEATURES))
        self.assertTrue(bool((bundle.gsr.values >= 0).all()))


if __name__ == "__main__":
    unittest.main()
