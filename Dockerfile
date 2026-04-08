ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies (with timeout fix)
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=300 uv sync

# 🔥 Force install everything correctly
RUN uv pip install openenv-core==0.2.2 numpy==1.26.4 torch==2.1.0

# Ensure torch is installed (fallback safety)
RUN uv pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

ENV PYTHONPATH="/app/env:$PYTHONPATH"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]