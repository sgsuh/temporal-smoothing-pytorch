FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# ffmpeg decodes the CamVid videos (Lagarith AVI, DVCPRO HD MXF) for frame extraction.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (and the package in editable mode) from a minimal copy of the
# sources so that this layer is cached until pyproject.toml changes. The full source
# tree is bind-mounted at /workspace at runtime (see docker-compose.yml).
COPY pyproject.toml README.md ./
COPY temporal_smoothing/__init__.py temporal_smoothing/__init__.py
RUN pip install -e ".[dev]"

# Allow running as a non-root host user (files created in the bind mount keep host ownership).
ENV HOME=/tmp
