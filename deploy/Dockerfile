# syntax=docker/dockerfile:1
#
# CPU image (default):
#   docker build -f deploy/Dockerfile -t jev-qwen:cpu .
#
# CUDA 12.8 image (requires the NVIDIA container runtime):
#   docker build -f deploy/Dockerfile \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
#     --build-arg TORCH_VERSION=2.11.0 \
#     -t jev-qwen:cuda .
#
# The image contains no weights. Mount a checkpoint directory at /models and
# set JEV_CHECKPOINT to the checkpoint subdirectory (or file) inside it.

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY requirements ./requirements
COPY src ./src

RUN python -m pip install --upgrade pip setuptools \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install -c requirements/constraints.txt -e ".[serve]" \
    && python -m pip check

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin jev \
    && mkdir -p /models /var/lib/jev-qwen \
    && chown -R jev:jev /models /var/lib/jev-qwen

USER jev

ENV JEV_CHECKPOINT=/models/checkpoint \
    JEV_LOG_LEVEL=INFO \
    JEV_METRICS_ENABLED=1 \
    HF_HOME=/var/lib/jev-qwen/hf

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "jev_qwen.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
