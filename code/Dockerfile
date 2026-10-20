# Version-locked environment for DiLT-CL v3.0.9.
#
# The image pins the exact Python packages recorded by every run behind the
# submitted results (requirements-lock.txt, taken from the runs' pip_freeze.txt).
# The GPU driver is not part of an image: the runs used NVIDIA driver 595.71.05
# with CUDA 13.0 on an RTX 5090. Bit-identical results are expected only on the
# same GPU architecture with the same libraries; see REPRODUCE.md.
#
# Build:  docker build -t diltcl:3.0.9 .
# Test:   docker run --rm diltcl:3.0.9 python -m pytest tests -q -p no:cacheprovider
# Use:    docker run --gpus all --rm -it -v /path/to/data:/data -v /path/to/runs:/runs diltcl:3.0.9
#
# The interpreter matches the recorded Python 3.12.3 (the runs used Anaconda's
# build of 3.12.3; this image uses the official CPython 3.12.3 build).
FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DILTCL_DATA_ROOT=/data \
    DILTCL_DOWNLOAD=0

WORKDIR /opt/diltcl
COPY requirements-lock.txt .
RUN python -m pip install --no-cache-dir -r requirements-lock.txt
COPY . .
CMD ["bash"]
