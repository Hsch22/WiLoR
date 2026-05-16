#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${WILOR_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE_TAG="${WILOR_IMAGE:-registry.mthreads.com/mcctest/ai/mtwan:4.3.3-pt2.7-v0.2}"
SHARED_IMAGE_LOADER="${WILOR_SHARED_IMAGE_LOADER:-/datapool/shared_images/mtwan_4.3.3-pt2.7-v0.2/load_mtwan.sh}"
PROXY_ENV="${WILOR_PROXY_ENV:-/datapool/.config/mihomo/proxy-env.sh}"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  bash "${SHARED_IMAGE_LOADER}"
fi

docker run --rm -i \
  --privileged \
  --network=host \
  --ipc=host \
  --runtime mthreads \
  -v /datapool:/datapool \
  -w "${PROJECT_ROOT}" \
  -e WILOR_BOOTSTRAP_IN_CONTAINER=1 \
  -e WILOR_PROJECT_ROOT="${PROJECT_ROOT}" \
  -e WILOR_PROXY_ENV="${PROXY_ENV}" \
  -e WILOR_VENV_DIR="${WILOR_VENV_DIR:-${PROJECT_ROOT}/.venv}" \
  -e WILOR_PYPI_MIRROR="${WILOR_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  -e WILOR_APT_PACKAGES="${WILOR_APT_PACKAGES:-libglvnd0 libegl1 libgl1 libglx0 libopengl0 libosmesa6}" \
  -e WILOR_SHARED_APT_CACHE="${WILOR_SHARED_APT_CACHE:-/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd}" \
  -e WILOR_INSTALL_APT_LIBS="${WILOR_INSTALL_APT_LIBS:-1}" \
  -e UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
  -e WILOR_HOST_UID="$(id -u)" \
  -e WILOR_HOST_GID="$(id -g)" \
  "${IMAGE_TAG}" \
  bash -lc '
    set -euo pipefail
    bash scripts/bootstrap_musa_uv.sh
    if [ -f "${WILOR_SHARED_APT_CACHE:-/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd}/env.sh" ]; then
      source "${WILOR_SHARED_APT_CACHE:-/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd}/env.sh"
    fi
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
    exec "$@"
  ' bash "$@"
