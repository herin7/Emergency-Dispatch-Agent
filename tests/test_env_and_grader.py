from __future__ import annotations

import importlib
import unittest

from emergency_dispatch.models import Action, ActionType
from emergency_dispatch.tasks import EasyDispatchTask, HardDispatchTask, MediumDispatchTask


class EmergencyDispatchEnvTests(unittest.TestCase):
    def test_easy_reset_matches_requested_defaults(self) -> None:
        env = EasyDispatchTask().create_env(seed=1)
        state = env.reset()

        self.assertEqual(state["grid_size"], 10)
        self.assertEqual(len(state["ambulances"]), 3)
        self.assertEqual(state["active_calls"], [])

    def test_step_returns_valid_shapes(self) -> None:
        env = HardDispatchTask().create_env(seed=7)
        env.reset()
        action = Action(action_type=ActionType.HOLD, ambulance_id="amb_0")

        state, reward, done, info = env.step(action)

        self.assertIn("ambulances", state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("metrics", info)
        self.assertIn("distance_matrix", state)

    def test_grader_score_is_bounded(self) -> None:
        task = HardDispatchTask()
        env = task.create_env(seed=7)
        state = env.reset()

        for _ in range(10):
            state, _, done, _ = env.step(env.heuristic_action())
            if done:
                break

        score = task.create_grader().grade(state).final_score
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_task_specs_expose_importable_graders(self) -> None:
        for task in [EasyDispatchTask(), MediumDispatchTask(), HardDispatchTask()]:
            spec = task.task_spec()
            module_name, function_name = spec["grader"].split(":")
            grader = getattr(importlib.import_module(module_name), function_name)
            score = grader()

            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
