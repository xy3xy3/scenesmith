import unittest

from scenesmith.agent_utils.manipuland_scale import (
    compute_uniform_scale_fit,
    diagnose_manipuland_scale,
    match_size_profile,
    normalize_manipuland_dimensions,
)


class TestManipulandScale(unittest.TestCase):
    """Tests for deterministic small-object scale helpers."""

    def test_match_size_profile(self) -> None:
        profile = match_size_profile("black desk lamp", "desk_lamp")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "desk_lamp")

    def test_normalize_known_manipuland_dimensions(self) -> None:
        dimensions, profile = normalize_manipuland_dimensions(
            description="spiral notebook",
            short_name="notebook",
            dimensions=[0.60, 0.04, 0.20],
        )

        self.assertEqual(profile.name, "notebook_book")
        self.assertEqual(dimensions, [0.30, 0.09, 0.06])

    def test_unknown_object_is_noop(self) -> None:
        dimensions, profile = normalize_manipuland_dimensions(
            description="small sculpture",
            short_name="sculpture",
            dimensions=[0.2, 0.2, 0.4],
        )

        self.assertIsNone(profile)
        self.assertEqual(dimensions, [0.2, 0.2, 0.4])

    def test_compute_uniform_scale_fit_rejects_bad_notebook_shape(self) -> None:
        fit = compute_uniform_scale_fit(
            current_dimensions=[0.107, 0.025, 0.143],
            desired_dimensions=[0.22, 0.16, 0.03],
            footprint_swappable=True,
        )

        self.assertGreater(fit.max_axis_relative_error, 0.75)

    def test_compute_uniform_scale_fit_allows_good_shape(self) -> None:
        fit = compute_uniform_scale_fit(
            current_dimensions=[0.20, 0.14, 0.02],
            desired_dimensions=[0.22, 0.16, 0.03],
            footprint_swappable=True,
        )

        self.assertLessEqual(fit.max_axis_relative_error, 0.75)

    def test_diagnose_manipuland_scale_reports_bad_fit(self) -> None:
        diagnostic = diagnose_manipuland_scale(
            description="notebook",
            actual_dimensions=[0.035, 0.047, 0.008],
            requested_dimensions=[0.22, 0.16, 0.03],
        )

        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic["profile"], "notebook_book")
        self.assertIn(
            diagnostic["status"], {"out_of_profile_range", "bad_uniform_scale_fit"}
        )


if __name__ == "__main__":
    unittest.main()
