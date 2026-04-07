FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install dependencies first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy application code
COPY . /app

# Healthcheck: verifies the environment can initialize
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "from emergency_dispatch.tasks import HardDispatchTask; t=HardDispatchTask(); e=t.create_env(seed=0); s=e.reset(); print('HEALTHY'); exit(0)" || exit 1

# Default: run inference (uses env vars for API keys)
CMD ["python", "inference.py"]
