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
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Default: run Gradio UI + OpenEnv HTTP server
CMD ["python", "app.py"]
