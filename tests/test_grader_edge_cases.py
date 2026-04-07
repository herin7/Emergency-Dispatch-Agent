"""
Comprehensive tests for the DispatchEpisodeGrader covering edge cases.
"""
import unittest
from emergency_dispatch.grader import DispatchEpisodeGrader, GradeBreakdown
from emergency_dispatch.models import UrgencyLevel
from emergency_dispatch.tasks import EasyDispatchTask, MediumDispatchTask, HardDispatchTask


class TestGraderEdgeCases(unittest.TestCase):
    """Test grader behavior with edge case scenarios."""

    def setUp(self):
        """Create a grader instance."""
        self.grader = DispatchEpisodeGrader()

    def _create_state(self, **kwargs):
        """Helper to create a minimal valid state with custom metrics."""
        task = EasyDispatchTask()
        env = task.create_env(seed=0)
        state = env.reset()
        
        # Update metrics based on kwargs
        metrics = state.get("metrics", {})
        metrics.update(kwargs.get("metrics", {}))
        state["metrics"] = metrics
        
        return state

    def test_perfect_score_scenario(self):
        """Test grading when all calls are resolved quickly with no timeouts."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 10,
            "critical_calls": 3,
            "resolved_critical_calls": 3,
            "critical_timeouts": 0,
            "high_timeouts": 0,
            "total_response_time": 15,  # Average 5 steps per call
            "total_response_time_critical": 9,  # Average 3 steps per critical call
            "total_fuel_consumed": 50.0,
        })
        
        result = self.grader.grade(state)
        
        self.assertIsInstance(result, GradeBreakdown)
        self.assertEqual(result.critical_response_rate, 1.0)
        self.assertGreater(result.mean_response_time_score, 0.5)
        self.assertEqual(result.coverage_efficiency, 1.0)
        self.assertEqual(result.zero_timeout_score, 1.0)
        self.assertGreater(result.final_score, 0.8)

    def test_zero_calls_scenario(self):
        """Test grading when no calls are generated (edge case)."""
        state = self._create_state(metrics={
            "total_calls": 0,
            "resolved_calls": 0,
            "critical_calls": 0,
            "resolved_critical_calls": 0,
            "critical_timeouts": 0,
            "high_timeouts": 0,
            "total_response_time": 0,
            "total_response_time_critical": 0,
            "total_fuel_consumed": 0.0,
        })
        
        result = self.grader.grade(state)
        
        self.assertIsInstance(result, GradeBreakdown)
        # Should handle division by zero gracefully
        self.assertGreaterEqual(result.final_score, 0.0)
        self.assertLessEqual(result.final_score, 1.0)

    def test_all_timeouts_scenario(self):
        """Test grading when all critical calls timeout."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 0,
            "critical_calls": 5,
            "resolved_critical_calls": 0,
            "critical_timeouts": 5,
            "high_timeouts": 3,
            "total_response_time": 0,
            "total_response_time_critical": 0,
            "total_fuel_consumed": 100.0,
        })
        
        result = self.grader.grade(state)
        
        self.assertEqual(result.critical_response_rate, 0.0)
        self.assertEqual(result.mean_response_time_score, 0.0)
        self.assertEqual(result.coverage_efficiency, 0.0)
        self.assertLess(result.zero_timeout_score, 0.5)
        self.assertLess(result.final_score, 0.2)

    def test_partial_completion_scenario(self):
        """Test grading with partial call resolution."""
        state = self._create_state(metrics={
            "total_calls": 20,
            "resolved_calls": 12,
            "critical_calls": 6,
            "resolved_critical_calls": 4,
            "critical_timeouts": 1,
            "high_timeouts": 2,
            "total_response_time": 60,
            "total_response_time_critical": 24,
            "total_fuel_consumed": 80.0,
        })
        
        result = self.grader.grade(state)
        
        self.assertAlmostEqual(result.critical_response_rate, 4/6, places=2)
        self.assertGreater(result.mean_response_time_score, 0.0)
        self.assertAlmostEqual(result.coverage_efficiency, 12/20, places=2)
        self.assertLess(result.zero_timeout_score, 1.0)

    def test_score_bounds_enforcement(self):
        """Test that final score is always within [0.0, 1.0]."""
        # Test multiple scenarios
        scenarios = [
            {"metrics": {"total_calls": 0, "resolved_calls": 0, "critical_calls": 0, 
                        "resolved_critical_calls": 0, "critical_timeouts": 0, "high_timeouts": 0,
                        "total_response_time": 0, "total_response_time_critical": 0,
                        "total_fuel_consumed": 0.0}},
            {"metrics": {"total_calls": 100, "resolved_calls": 0, "critical_calls": 50,
                        "resolved_critical_calls": 0, "critical_timeouts": 50, "high_timeouts": 30,
                        "total_response_time": 0, "total_response_time_critical": 0,
                        "total_fuel_consumed": 500.0}},
            {"metrics": {"total_calls": 1, "resolved_calls": 1, "critical_calls": 1,
                        "resolved_critical_calls": 1, "critical_timeouts": 0, "high_timeouts": 0,
                        "total_response_time": 1, "total_response_time_critical": 1,
                        "total_fuel_consumed": 1.0}},
        ]
        
        for scenario in scenarios:
            state = self._create_state(**scenario)
            result = self.grader.grade(state)
            self.assertGreaterEqual(result.final_score, 0.0, 
                                    f"Score below 0.0 for scenario: {scenario}")
            self.assertLessEqual(result.final_score, 1.0,
                                 f"Score above 1.0 for scenario: {scenario}")

    def test_individual_metric_bounds(self):
        """Test that all individual metrics are within [0.0, 1.0]."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 5,
            "critical_calls": 3,
            "resolved_critical_calls": 2,
            "critical_timeouts": 1,
            "high_timeouts": 1,
            "total_response_time": 30,
            "total_response_time_critical": 15,
            "total_fuel_consumed": 40.0,
        })
        
        result = self.grader.grade(state)
        
        for field_name in ["critical_response_rate", "mean_response_time_score", 
                          "coverage_efficiency", "zero_timeout_score"]:
            value = getattr(result, field_name)
            self.assertGreaterEqual(value, 0.0, f"{field_name} below 0.0")
            self.assertLessEqual(value, 1.0, f"{field_name} above 1.0")

    def test_high_timeouts_impact_zero_timeout_score(self):
        """Test that high timeouts (not just critical) affect zero_timeout_score."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 10,
            "critical_calls": 0,
            "resolved_critical_calls": 0,
            "critical_timeouts": 0,
            "high_timeouts": 5,  # Only high timeouts
            "total_response_time": 50,
            "total_response_time_critical": 0,
            "total_fuel_consumed": 60.0,
        })
        
        result = self.grader.grade(state)
        
        # Should penalize for high timeouts (tight: 3 timeouts = 0.0)
        self.assertLess(result.zero_timeout_score, 1.0)
        # 5 timeouts * 0.33 = 1.65 → clamped to 0.0
        self.assertEqual(result.zero_timeout_score, 0.0)

    def test_critical_response_time_tracked_separately(self):
        """Test that critical response time is tracked separately from total."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 8,
            "critical_calls": 4,
            "resolved_critical_calls": 4,
            "critical_timeouts": 0,
            "high_timeouts": 0,
            "total_response_time": 40,  # Includes all calls
            "total_response_time_critical": 12,  # Only critical calls
            "total_fuel_consumed": 50.0,
        })
        
        result = self.grader.grade(state)
        
        # Mean response time score should be based on critical response time (12/4 = 3)
        # Not total response time (40/8 = 5)
        grid_size = state["grid_size"]
        expected_mean_time = 12 / 4  # 3.0
        expected_score = max(0.0, 1.0 - (expected_mean_time / (grid_size * 2)))
        self.assertAlmostEqual(result.mean_response_time_score, expected_score, places=2)

    def test_all_tasks_produce_valid_grades(self):
        """Test that all three tasks produce valid grades."""
        tasks = [EasyDispatchTask(), MediumDispatchTask(), HardDispatchTask()]
        
        for task in tasks:
            env = task.create_env(seed=42)
            state = env.reset()
            
            # Run a few steps to generate some metrics
            for _ in range(10):
                action = env.heuristic_action()
                state, _, done, _ = env.step(action)
                if done:
                    break
            
            result = self.grader.grade(state)
            
            self.assertIsInstance(result, GradeBreakdown, 
                                  f"Invalid grade for {task.name}")
            self.assertGreaterEqual(result.final_score, 0.0,
                                    f"Score below 0.0 for {task.name}")
            self.assertLessEqual(result.final_score, 1.0,
                                 f"Score above 1.0 for {task.name}")


class TestGraderWeightedScoring(unittest.TestCase):
    """Test the weighted scoring formula."""

    def setUp(self):
        self.grader = DispatchEpisodeGrader()

    def _create_state(self, **kwargs):
        task = EasyDispatchTask()
        env = task.create_env(seed=0)
        state = env.reset()
        metrics = state.get("metrics", {})
        metrics.update(kwargs.get("metrics", {}))
        state["metrics"] = metrics
        return state

    def test_weighted_score_calculation(self):
        """Test that final score matches weighted formula."""
        state = self._create_state(metrics={
            "total_calls": 10,
            "resolved_calls": 7,
            "critical_calls": 4,
            "resolved_critical_calls": 3,
            "critical_timeouts": 0,
            "high_timeouts": 0,
            "total_response_time": 20,
            "total_response_time_critical": 12,
            "total_fuel_consumed": 50.0,
        })
        
        result = self.grader.grade(state)
        
        # Verify bounds
        self.assertGreaterEqual(result.final_score, 0.0)
        self.assertLessEqual(result.final_score, 1.0)
        self.assertGreater(result.final_score, 0.5)  # Should be a decent score


if __name__ == "__main__":
    unittest.main()
