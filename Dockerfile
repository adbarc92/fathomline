# The gate this project ships against is "one command on a clean container
# reproduces a published eval result", so the container exists from the first
# commit rather than arriving as packaging work at the end. A harness that only
# runs on the author's machine has failed its own thesis.
#
# Spike option C (docs/SPIKE-week1-policy.md): SmolVLA + LIBERO on LeRobot 0.6.1,
# Python 3.12, CPU torch, every artifact baked in so an eval needs no network.
#
# Layers run from slowest-changing to fastest, so a code change rebuilds only the
# last one and the GHCR cache (CI `container` job) serves the rest:
#   system libraries -> locked dependencies -> pinned Hub snapshots -> source.

# Base images pinned by digest: a tag is a promise, a digest is a fact.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

COPY --from=ghcr.io/astral-sh/uv:0.7.13@sha256:6c1e19020ec221986a210027040044a5df8de762eb36d5240e382bc41d7a9043 /uv /bin/

LABEL org.opencontainers.image.source="https://github.com/adbarc92/fathomline"

# MUJOCO_GL / PYOPENGL_PLATFORM: headless MuJoCo through EGL on Mesa's llvmpipe.
# The venv is on PATH and the source on PYTHONPATH, so `python` is the project's
# interpreter without `uv run` re-resolving anything at container start.
ENV UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    HF_HOME=/opt/hf \
    FATHOMLINE_ARTIFACTS=/opt/fathomline \
    LIBERO_CONFIG_PATH=/opt/fathomline/libero

# EGL and Mesa for rendering. glib and xcb for `opencv-python`: something in
# LIBERO's tree pulls the non-headless build, and it overwrites the headless one
# LeRobot asks for, so `import cv2` needs them (first CI run: missing libgthread).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libegl-mesa0 libgl1-mesa-dri libglib2.0-0t64 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --locked: the lock file is the contract. If it is stale against pyproject.toml,
# fail here rather than quietly building something different from what CI tested.
# (--frozen would install the lock without checking it.)
# The build tools are for `egl-probe` and `hf-egl-probe`: sdists only, whose
# setup.py shells out to a system `cmake` and `make` to build a C extension.
# They are removed in the same layer, so they cost the image nothing.
COPY pyproject.toml uv.lock ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ make cmake \
    && uv sync --locked --no-dev --extra lerobot --no-install-project \
    && apt-get purge -y --auto-remove gcc g++ make cmake \
    && rm -rf /var/lib/apt/lists/*

# Only the files warm.py imports, so source changes elsewhere keep this layer.
COPY src/fathomline/__init__.py src/fathomline/
COPY src/fathomline/adapters/__init__.py src/fathomline/adapters/
COPY src/fathomline/adapters/lerobot/__init__.py \
     src/fathomline/adapters/lerobot/pins.py \
     src/fathomline/adapters/lerobot/warm.py \
     src/fathomline/adapters/lerobot/
# Downloads every pinned snapshot at its revision SHA and fails the build if any
# does not hash to its pinned digest.
RUN python -m fathomline.adapters.lerobot.warm

# From here on, anything unpinned that reaches for the Hub fails instead of drifting.
ENV HF_HUB_OFFLINE=1

COPY src ./src

ENTRYPOINT ["python", "-m", "fathomline"]
CMD ["--help"]
