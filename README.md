# Emergency Dispatch Agent

Emergency Dispatch Agent is an OpenEnv-style reinforcement learning environment for ambulance fleet coordination. The project simulates a real-world dispatch workflow where ambulances move across a city grid, new emergency calls appear dynamically, and the agent must balance urgency, fuel, and coverage under uncertainty.

## Why This Environment

This project models a real operational problem instead of a toy game. It is designed around:

- dynamic emergency arrivals
- multi-ambulance resource allocation
- urgency-aware dispatch decisions
- fuel constraints and return-to-base logistics
- deterministic grading with bounded scores from `0.0` to `1.0`

## Observation Space

The environment state includes:

- city grid as an `NxN` integer matrix
- ambulance positions
- ambulance fuel levels
- ambulance status values
- active emergency calls
- completed calls
- current step count
- episode metrics used by the grader

Typed models are implemented with Pydantic in `emergency_dispatch/models.py`.

## Action Space

Supported actions:

- `dispatch`
- `return_to_base`
- `hold`
- `reassign`

Each action is validated through a typed Pydantic model before being applied to the simulation.

## Tasks

Three tasks are included:

- `EasyDispatchTask`: `10x10` grid, `3` ambulances, mostly low and medium urgency calls
- `MediumDispatchTask`: `15x15` grid, `5` ambulances, mixed urgency with stronger fuel pressure
- `HardDispatchTask`: `20x20` grid, `5` ambulances, frequent critical calls

## Reward Design

The reward function provides dense trajectory feedback:

- positive rewards for resolving calls quickly
- larger rewards for critical and high-priority calls
- penalties for critical and high-priority timeouts
- penalty for invalid dispatches
- penalty for fuel exhaustion mid-dispatch
- per-step efficiency cost

## Grading

The grader combines:

- Critical Response Rate: `40%`
- Mean Response Time: `25%`
- Coverage Efficiency: `15%`
- Zero Timeout Score: `20%`

Final scores are clipped to the range `0.0` to `1.0`.

## Inference

The submission runner is `inference.py`.

It:

- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- uses the OpenAI client for LLM calls
- runs one complete reproducible episode
- emits strict `[START]`, `[STEP]`, and `[END]` logs

Optional variables:

- `LOCAL_IMAGE_NAME`
- `EMERGENCY_DISPATCH_TASK` with values `easy`, `medium`, or `hard`

## Setup

```powershell
cd C:\Users\Admin\Desktop\Development\metaverse
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
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
$env:HF_TOKEN='<your-token>'
python inference.py
```

## Docker

```powershell
docker build -t emergency-dispatch .
docker run --rm -e API_BASE_URL='http://host.docker.internal:8765' -e MODEL_NAME='mock-model' -e HF_TOKEN='mock-token' emergency-dispatch
```

## Project Docs

- `mission.md`: project motivation and problem statement
- `info.md`: detailed technical explanation
- `run.md`: step-by-step run instructions
