# Emergency Dispatch Agent

Emergency Dispatch Agent is an OpenEnv-compatible environment for ambulance fleet coordination. It models a real dispatch workflow where an agent must assign ambulances to incoming emergencies while balancing urgency, fuel, travel distance, and city coverage.

## Why This Environment

- Real-world task: emergency response coordination is an operational problem humans solve under pressure.
- Dense rewards: the agent receives partial credit for progress and penalties for bad dispatch decisions.
- Deterministic grading: each task has a bounded grader score in `[0.0, 1.0]`.
- Difficulty progression: the environment includes `easy`, `medium`, and `hard` tasks.

## Observation Space

The environment state includes:

```json
{
  "step_count": 0,
  "grid_size": 10,
  "ambulances": [
    {
      "id": "amb_0",
      "x": 0,
      "y": 0,
      "fuel_level": 100.0,
      "status": "idle",
      "base_x": 0,
      "base_y": 0,
      "target_x": null,
      "target_y": null,
      "assigned_call_id": null,
      "dispatch_start_step": null
    }
  ],
  "active_calls": [],
  "completed_calls": [],
  "cumulative_reward": 0.0,
  "max_steps": 200,
  "metrics": {
    "total_calls": 0,
    "resolved_calls": 0,
    "critical_calls": 0,
    "resolved_critical_calls": 0,
    "critical_timeouts": 0,
    "high_timeouts": 0,
    "total_response_time": 0,
    "total_response_time_critical": 0,
    "movement_steps": 0,
    "useful_steps": 0,
    "fuel_out_events": 0,
    "total_fuel_consumed": 0.0
  },
  "distance_matrix": {},
  "mode": "running"
}
```

Typed models are implemented with Pydantic in `emergency_dispatch/models.py`.

## Action Space

Supported actions:

| Action | Description | Required Fields |
|--------|-------------|-----------------|
| `dispatch` | Assign an idle or holding ambulance to a call | `ambulance_id`, `call_id` |
| `reassign` | Redirect an already-dispatched ambulance to a new call | `ambulance_id`, `call_id` |
| `return_to_base` | Send an ambulance home to refuel | `ambulance_id` |
| `hold` | Keep an ambulance stationary | `ambulance_id` |

Action payloads are validated by the `Action` Pydantic model before they are applied.

## Tasks

The environment ships with three tasks and three importable graders:

| Task ID | Class | Description | Grader |
|---------|-------|-------------|--------|
| `easy` | `EasyDispatchTask` | 10x10 grid, 3 ambulances, mostly low and medium calls | `emergency_dispatch.grader:grade_easy` |
| `medium` | `MediumDispatchTask` | 15x15 grid, 5 ambulances, mixed calls with stronger fuel pressure | `emergency_dispatch.grader:grade_medium` |
| `hard` | `HardDispatchTask` | 20x20 grid, 5 ambulances, frequent critical calls | `emergency_dispatch.grader:grade_hard` |

The task manifest is declared in `openenv.yaml`.

## Reward Design

The reward is dense over the full episode:

- Positive rewards for resolving calls, with larger bonuses for critical and high-urgency calls.
- Per-step cost to discourage wasting time.
- Penalties for invalid dispatches, fuel exhaustion, and timed-out calls.
- Task difficulty changes call frequency, grid size, and urgency mix.

## Grading

Each task is scored with the same deterministic grading framework:

- Critical response rate
- Mean critical response time score
- Coverage efficiency
- Timeout avoidance
- Task-specific objective score

The final score is clipped to `[0.0, 1.0]`.

## Baseline Scores

The heuristic baseline in `tmp_baseline.py` produces the following mean scores across seeds `7, 42, 99, 123, 256`:

| Task | Mean Score | Std Dev |
|------|------------|---------|
| `easy` | 0.7441 | 0.2384 |
| `medium` | 0.6634 | 0.0710 |
| `hard` | 0.3559 | 0.0071 |

## Baseline Inference

The submission runner is `inference.py`.

It:

- Uses the OpenAI client for all model calls
- Reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment
- Runs `easy`, `medium`, and `hard` sequentially by default
- Emits one strict `[START]` -> `[STEP]` -> `[END]` log sequence per task ID
- Falls back to the heuristic policy if the model request fails

Optional selector:

```bash
EMERGENCY_DISPATCH_TASK=all
EMERGENCY_DISPATCH_TASK=easy
EMERGENCY_DISPATCH_TASK=medium
EMERGENCY_DISPATCH_TASK=hard
```

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install .
```

## Run Tests

```powershell
python -m unittest discover -s tests -v
```

## Run Inference With Mock API

Terminal 1:

```powershell
python tests\mock_openai_server.py
```

Terminal 2:

```powershell
$env:API_BASE_URL='http://127.0.0.1:8765'
$env:MODEL_NAME='mock-model'
$env:HF_TOKEN='mock-token'
python inference.py
```

## Run Inference With Real API

```powershell
$env:API_BASE_URL='https://router.huggingface.co/v1'
$env:MODEL_NAME='Qwen/Qwen2.5-72B-Instruct'
$env:HF_TOKEN='your-token-here'
python inference.py
```

## Docker

Build:

```powershell
docker build -t emergency-dispatch .
```

Run the Space server:

```powershell
docker run --rm -p 7860:7860 emergency-dispatch
```

Run inference in the container:

```powershell
docker run --rm `
  -e API_BASE_URL='https://router.huggingface.co/v1' `
  -e MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' `
  -e HF_TOKEN='your-token' `
  emergency-dispatch python inference.py
```

## Hugging Face Space Deployment

1. Create a Docker Space tagged with `openenv`.
2. Push this repository to the Space.
3. Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` in the Space settings.
4. The container will start `app.py`, which serves `/reset`, `/step`, `/state`, `/tasks`, and grading endpoints.

## Project Files

- `app.py`: FastAPI and Gradio entrypoint
- `inference.py`: baseline runner
- `openenv.yaml`: task and environment manifest
- `emergency_dispatch/env.py`: environment logic
- `emergency_dispatch/tasks.py`: task definitions
- `emergency_dispatch/grader.py`: grader functions and scoring logic
