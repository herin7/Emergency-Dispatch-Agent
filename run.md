# Emergency Dispatch Agent - Run Guide

## Environment Variables

Set the model credentials before running inference:

```bash
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export EMERGENCY_DISPATCH_TASK="all"
```

`EMERGENCY_DISPATCH_TASK` can also be `easy`, `medium`, or `hard`.

## Local Server

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install .
python app.py
```

Health check:

```powershell
curl http://localhost:7860/health
```

## Baseline Inference

The submission runner is the root-level `inference.py` script. By default it runs `easy`, `medium`, and `hard` sequentially and emits strict `[START]`, `[STEP]`, and `[END]` logs for each task.

```powershell
python inference.py
```

## Docker

```powershell
docker build -t emergency-dispatch .
docker run --rm -p 7860:7860 emergency-dispatch
```

Run the baseline in Docker:

```powershell
docker run --rm `
  -e API_BASE_URL='https://router.huggingface.co/v1' `
  -e MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' `
  -e HF_TOKEN='your-token' `
  emergency-dispatch python inference.py
```

## Validator Checklist

- `openenv.yaml` declares 3 task IDs with importable graders
- `/tasks` returns the same 3 tasks and grader references
- `inference.py` logs `task=easy`, `task=medium`, and `task=hard`
- scores are normalized to `[0.0, 1.0]`
