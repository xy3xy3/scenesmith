import unittest

from scenesmith.agent_utils.clearance_zones import WindowClearanceViolation
from scenesmith.agent_utils.physics_tools import _build_violation_message
from scenesmith.agent_utils.physics_validation import CollisionPair


class TestBuildViolationMessage(unittest.TestCase):
    def test_window_only_warnings_are_advisory(self):
        result = _build_violation_message(
            collisions=[],
            thin_covering_overlaps=[],
            thin_covering_boundary_violations=[],
            door_violations=[],
            open_violations=[],
            height_violations=[],
            window_violations=[
                WindowClearanceViolation(
                    furniture_id="dresser_1",
                    window_label="window_2",
                    furniture_top_height=1.2,
                    sill_height=0.9,
                )
            ],
        )

        self.assertIn("No physics violations detected", result)
        self.assertIn("Window access warnings:", result)
        self.assertIn("treat them as advisory", result)
        self.assertNotIn("Physics violations detected", result)
        self.assertNotIn("Please resolve", result)

    def test_window_warnings_do_not_count_as_blocking_issues(self):
        result = _build_violation_message(
            collisions=[
                CollisionPair(
                    object_a_name="chair",
                    object_a_id="chair_1",
                    object_b_name="table",
                    object_b_id="table_1",
                    penetration_depth=0.05,
                )
            ],
            thin_covering_overlaps=[],
            thin_covering_boundary_violations=[],
            door_violations=[],
            open_violations=[],
            height_violations=[],
            window_violations=[
                WindowClearanceViolation(
                    furniture_id="dresser_1",
                    window_label="window_2",
                    furniture_top_height=1.2,
                    sill_height=0.9,
                )
            ],
        )

        self.assertIn("Physics violations detected (1 issue(s))", result)
        self.assertIn("Collisions (1):", result)
        self.assertIn("Window access warnings:", result)
        self.assertNotIn("Physics violations detected (2 issue(s))", result)


if __name__ == "__main__":
    unittest.main()
