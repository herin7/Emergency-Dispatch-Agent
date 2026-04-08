# Emergency Dispatch Agent - Run Guide

This guide ensures that the project runs properly according to the **OpenEnv Hackathon (Round 1) Requirements**, making use of Docker, the custom HuggingFace Space UI, and the Baseline inference script.

---

## 1. Environment Variables Configuration

Before you start, make sure you configure your credentials correctly. You must supply an API key that your LLM provider expects or defaults. 
If using HuggingFace serverless inference format, `HF_TOKEN` must be present. You can export these to your terminal, or place them in an `.env` file for Docker.

```bash
# Set up your environment variables
export OPENAI_API_KEY="your-openai-api-key"   # Or set this
export HF_TOKEN="your-hf-token"               # Or set this
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export EMERGENCY_DISPATCH_TASK="medium" # Set to "easy", "medium", or "hard". Defaults to "hard".
```

---

## 2. Option A: Run the Gradio UI & OpenEnv APIs Locally

As required by the hackathon, the platform provides a Gradio Hub along with the automated OpenEnv API endpoints (`/reset`, `/step`, `/state`). This server binds on port `7860`.

```powershell
# Create Virtual Environment & Install required packages
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start the interactive UI and server
python app.py
```
> **Verify** it runs successfully by:
> 1. Opening `http://localhost:7860/` for the UI
> 2. Pinging `http://localhost:7860/health` or `http://localhost:7860/reset` (POST route via curl)

---

## 3. Option B: Run Baseline Inference Script (`inference.py`)

The submission provides the mandatory `inference.py` located at the root of the project. It explicitly runs a chosen scenario completely using the standard OpenAI client. Time limit per run must stay **below 20 minutes** on hackathon machines.

1. Ensure requirements are installed and virtualenv is activated.
2. Launch the inference sequence (uses environment variables automatically):
```powershell
python inference.py
```

It outputs logs identically matching the explicit `[START]`, `[STEP]`, `[END]` hackathon format. No other custom stdout is emitted.

---

## 4. Option C: Run via Docker (Validation Ready)

The environment must be containerized to operate successfully on Hugging Face Spaces. It uses a clean Python `3.11-slim` profile matching system requirements to operate within **vCPU: 2, Data RAM: 8GB**. 

**1. Create the Docker Image:**
```powershell
docker build -t emergency-dispatch .
```

**2. Start the Deployment Container:**
```powershell
# By default, Docker executes app.py starting Gradio and the API routes. 
docker run -p 7860:7860 --rm -e API_BASE_URL='https://router.huggingface.co/v1' -e MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' -e HF_TOKEN='your-valid-token' -e OPENAI_API_KEY='your-valid-key' emergency-dispatch
```

**3. Run the CLI Inference within Docker:**
```powershell
docker run --rm -e API_BASE_URL='https://router.huggingface.co/v1' -e MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' -e HF_TOKEN='your-valid-token' emergency-dispatch python inference.py
```

---

## Summary Checklist for Submission

All the criteria from the Hackathon Round 1 are fulfilled:
✅ Standard typed actions, observations, and rewards built with Pydantic compliant with `OpenEnv` constraints.
✅ Fully configured API handling including endpoints `step()`, `reset()`, and `state()`.
✅ Contains `openenv.yaml`.
✅ 3 distinct difficulty tasks (`EasyDispatchTask`, `MediumDispatchTask`, `HardDispatchTask`) scaling progressively.
✅ Zero static-grade scoring metric. Real performance impacts grid logic.
✅ Works natively via Docker deployments on 2vCPUs, 8GB setup on HF platforms tagged `openenv`.
✅ Contains a fully spec'd `inference.py` running underneath **20 mins** timeout strict parsing.
