FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app/data/home \
    HF_HOME=/app/data/cache/huggingface \
    PADDLE_PDX_CACHE_HOME=/app/data/cache/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    XDG_CACHE_HOME=/app/data/cache \
    HF_ENDPOINT=https://huggingface.co

WORKDIR /app
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o APT::Update::Error-Mode=any \
    && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY infra/deploy/requirements-server.txt infra/deploy/requirements-server.txt
# Install CPU wheels first so the regular requirements do not pull CUDA packages.
RUN python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r infra/deploy/requirements-server.txt \
    && python -m pip check

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir -p /app/data/home /app/data/cache /app/data/models \
    && chown -R app:app /app/data
COPY back back
COPY scripts scripts
COPY resources resources
COPY infra/deploy infra/deploy
USER 10001:10001
EXPOSE 8000
# One worker: SQLite and model instances are shared within this process.
CMD ["python", "-m", "uvicorn", "back.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
