# Self-contained Dockerfile for serving a PI policy on a remote machine.
# Unlike serve_policy.Dockerfile, this bakes the source code into the image
# so no volume mounts are needed — suitable for deployment on remote machines.
#
# Build:
#   docker build . -t openpi_server -f scripts/docker/serve_policy_standalone.Dockerfile
#
# Run (weights downloaded from GCS on first start):
#   docker run --rm -it --gpus=all -p 8000:8000 \
#     -v ~/.cache/openpi:/openpi_assets \
#     -e SERVER_ARGS="--env droid" \
#     openpi_server

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

# Copy dependency manifests first so Docker can cache the install layer.
COPY uv.lock pyproject.toml LICENSE README.md ./
COPY packages/ packages/

RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-project

# Copy source and scripts.
COPY src/ src/
COPY scripts/ scripts/

# Install the openpi project itself now that src/ is present.
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev

# Apply the transformers monkey-patches that the PyTorch models require.
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r src/openpi/models_pytorch/transformers_replace/* {}

ENV OPENPI_DATA_HOME=/openpi_assets

EXPOSE 8000

CMD /bin/bash -c "/.venv/bin/python scripts/serve_policy.py $SERVER_ARGS"
