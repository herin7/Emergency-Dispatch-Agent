# How To Run Emergency Dispatch Agent

## Overview

This document explains how to run the Emergency Dispatch Agent project locally for:

- installation
- environment testing
- inference testing with a mock API
- inference testing with a real API endpoint
- Docker execution

The project is designed for Python `3.11`, but it has also been smoke-tested locally with Python `3.12`.

## Project Files You Will Use

Main files:

- `inference.py`
- `openenv.yaml`
- `requirements.txt`
- `Dockerfile`

Main package:

- `emergency_dispatch/`

Tests:

- `tests/test_env_and_grader.py`
- `tests/test_inference_format.py`
- `tests/mock_openai_server.py`

## 1. Open the Project Folder

In PowerShell:

```powershell
cd C:\Users\Admin\Desktop\Development\metaverse
```

## 2. Create and Activate a Virtual Environment

Recommended with Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If Python 3.11 is not available, you can use another installed Python version:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

This installs:

- `gymnasium`
- `pydantic`
- `numpy`
- `openai`
- `huggingface_hub`

## 4. Run the Automated Tests

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

What this checks:

- environment reset behavior
- step output structure
- grader score bounds
- `inference.py` output format

If the tests pass, you should see:

```text
Ran 4 tests in ...
OK
```

## 5. Run a Quick Environment Smoke Test

You can manually step through the environment from the terminal.

```powershell
@'
from emergency_dispatch.tasks import HardDispatchTask

task = HardDispatchTask()
env = task.create_env(seed=7)
state = env.reset()

for _ in range(5):
    action = env.heuristic_action()
    state, reward, done, info = env.step(action)
    print("reward =", reward, "done =", done)
    if done:
        break

print("final mode =", state["mode"])
'@ | python -
```

This runs a few steps using the built-in heuristic policy.

## 6. Test inference.py Without Spending API Quota

The repository includes a local mock OpenAI-compatible server so you can validate the full inference pipeline without calling a real hosted model.

### Step A: Start the mock server

In one PowerShell window:

```powershell
python tests\mock_openai_server.py
```

This starts a local server at:

```text
http://127.0.0.1:8765
```

### Step B: Run inference.py in a second terminal

```powershell
$env:API_BASE_URL='http://127.0.0.1:8765'
$env:MODEL_NAME='mock-model'
$env:HF_TOKEN='mock-token'
python inference.py
```

Expected output shape:

```text
[START]
[STEP] State: {...} Action: {...} Reward: ...
[STEP] State: {...} Action: {...} Reward: ...
[END] Score: ...
```

This is the safest way to verify:

- environment variable reading
- OpenAI client wiring
- output tag format
- episode execution

## 7. Run inference.py With a Real API Endpoint

When you want to test against a real provider, set the environment variables provided by the platform or your own endpoint.

Example:

```powershell
$env:API_BASE_URL='https://your-openai-compatible-endpoint/v1'
$env:MODEL_NAME='your-model-name'
$env:HF_TOKEN='your-api-key-or-token'
python inference.py
```

Important notes:

- do not hardcode keys into the code
- judges will provide their own environment variables during evaluation
- the script runs one full episode only
- the script falls back to a local heuristic if the API fails or is rate-limited

## 8. Understand What inference.py Does

`inference.py` performs one reproducible evaluation run on `HardDispatchTask`.

Flow:

1. Reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
2. Creates the OpenAI client
3. Creates the `HardDispatchTask` environment
4. Calls `reset()`
5. Prints `[START]`
6. Repeats:
   - sends current state JSON to the model
   - parses the returned action JSON
   - calls `env.step(action)`
   - prints one `[STEP]` line
7. Grades the final state
8. Prints `[END] Score: ...`

## 9. Run With Docker

Build the image:

```powershell
docker build -t emergency-dispatch .
```

Run it with environment variables:

```powershell
docker run --rm `
  -e API_BASE_URL='http://host.docker.internal:8765' `
  -e MODEL_NAME='mock-model' `
  -e HF_TOKEN='mock-token' `
  emergency-dispatch
```

If you are connecting the container to a real hosted endpoint, replace `API_BASE_URL` with your actual provider URL.

## 10. Files Used by OpenEnv-Style Execution

The main environment configuration is:

```text
openenv.yaml
```

The important entrypoints are:

- simulation: `emergency_dispatch.env:EmergencyDispatchEnv`
- grader: `emergency_dispatch.grader:DispatchEpisodeGrader`
- tasks:
  - `emergency_dispatch.tasks:EasyDispatchTask`
  - `emergency_dispatch.tasks:MediumDispatchTask`
  - `emergency_dispatch.tasks:HardDispatchTask`

## 11. Common Issues and Fixes

### Python not found

If `python` is not recognized, try:

```powershell
py -3.11 --version
```

or:

```powershell
py -3.12 --version
```

Then use that interpreter to create the virtual environment.

### Missing dependencies

If imports fail, reinstall requirements:

```powershell
pip install -r requirements.txt
```

### Port 8765 already in use

If the mock server fails to start, stop the process using that port or change the port in:

- `tests/mock_openai_server.py`
- your `API_BASE_URL` environment variable

### Real API rate limits

`inference.py` already uses short timeouts and limited retries. If the model provider is unstable, the script will fall back to the heuristic policy instead of crashing.

## 12. Recommended Validation Flow

Best order for checking everything:

1. Create and activate the virtual environment
2. Install `requirements.txt`
3. Run `python -m unittest discover -s tests -v`
4. Run the mock server
5. Run `python inference.py` against the mock server
6. Run `python inference.py` against the real endpoint
7. Build and test the Docker image

## 13. Minimal Commands Summary

### Setup

```powershell
cd C:\Users\Admin\Desktop\Development\metaverse
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Tests

```powershell
python -m unittest discover -s tests -v
```

### Mock inference

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

### Docker

```powershell
docker build -t emergency-dispatch .
docker run --rm -e API_BASE_URL='http://host.docker.internal:8765' -e MODEL_NAME='mock-model' -e HF_TOKEN='mock-token' emergency-dispatch
```

## Final Note

If your goal is hackathon submission readiness, the fastest reliable check is:

- run the tests
- run `inference.py` against the mock server
- run `inference.py` once against the real endpoint
- confirm the output preserves `[START]`, `[STEP]`, and `[END] Score: ...`

That will give you strong confidence that the environment and inference pipeline are ready for evaluation.
