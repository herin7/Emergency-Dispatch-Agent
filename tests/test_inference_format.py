from __future__ import annotations

import json
import os
import subprocess
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe")


class InferenceFormatTests(unittest.TestCase):
    def test_inference_emits_required_log_format(self) -> None:
        server = subprocess.Popen(
            [str(PYTHON), str(ROOT / "tests" / "mock_openai_server.py")],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.0)
            env = os.environ.copy()
            env["API_BASE_URL"] = "http://127.0.0.1:8765"
            env["MODEL_NAME"] = "mock-model"
            env["HF_TOKEN"] = "mock-token"

            result = subprocess.run(
                [str(PYTHON), str(ROOT / "inference.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                check=True,
            )
        finally:
            server.terminate()
            server.wait(timeout=5)

        lines = [line for line in result.stdout.strip().splitlines() if line]
        self.assertGreaterEqual(len(lines), 3)
        self.assertEqual(lines[0], "[START] task=HardDispatchTask env=EmergencyDispatch model=mock-model")
        self.assertTrue(lines[-1].startswith("[END] success="))

        first_step = next(line for line in lines if line.startswith("[STEP] "))
        self.assertIn(" step=1 ", first_step)
        self.assertIn(" action=", first_step)
        self.assertIn(" reward=", first_step)
        self.assertIn(" done=", first_step)
        self.assertIn(" error=", first_step)

        action_fragment = first_step.split(" action=", 1)[1].split(" reward=", 1)[0]
        reward_fragment = first_step.split(" reward=", 1)[1].split(" done=", 1)[0]
        done_fragment = first_step.split(" done=", 1)[1].split(" error=", 1)[0]
        error_fragment = first_step.split(" error=", 1)[1]

        action = json.loads(action_fragment)
        reward = float(reward_fragment)

        self.assertIn("action_type", action)
        self.assertIsInstance(reward, float)
        self.assertIn(done_fragment, {"true", "false"})
        self.assertTrue(error_fragment == "null" or isinstance(error_fragment, str))

        end_line = lines[-1]
        score_fragment = end_line.split(" score=", 1)[1].split(" rewards=", 1)[0]
        rewards_fragment = end_line.split(" rewards=", 1)[1]
        score = float(score_fragment)
        reward_values = [float(value) for value in rewards_fragment.split(",")] if rewards_fragment else []

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(len(reward_values), 1)


if __name__ == "__main__":
    unittest.main()
