#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${WILOR_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${WILOR_VENV_DIR:-${PROJECT_ROOT}/.venv}"
IMAGE_TAG="${WILOR_IMAGE:-registry.mthreads.com/mcctest/ai/mtwan:4.3.3-pt2.7-v0.2}"
SHARED_IMAGE_LOADER="${WILOR_SHARED_IMAGE_LOADER:-/datapool/shared_images/mtwan_4.3.3-pt2.7-v0.2/load_mtwan.sh}"
PYPI_MIRROR="${WILOR_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PROXY_ENV="${WILOR_PROXY_ENV:-/datapool/.config/mihomo/proxy-env.sh}"
LOCK_FILE="${WILOR_VENV_LOCK_FILE:-${VENV_DIR}.lock}"
APT_PACKAGES=(${WILOR_APT_PACKAGES:-libglvnd0 libegl1 libgl1 libglx0 libopengl0 libosmesa6})
SHARED_APT_CACHE="${WILOR_SHARED_APT_CACHE:-/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd}"
STAMP_FILE="${VENV_DIR}/.wilor-musa-env.sha256"

in_mtwan_image() {
  python3 - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(1)
if importlib.util.find_spec("torch_musa") is None:
    raise SystemExit(1)
PY
}

run_in_container() {
  if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    bash "${SHARED_IMAGE_LOADER}"
  fi

  docker run --rm \
    --privileged \
    --network=host \
    --ipc=host \
    --runtime mthreads \
    -v /datapool:/datapool \
    -w "${PROJECT_ROOT}" \
    -e WILOR_BOOTSTRAP_IN_CONTAINER=1 \
    -e WILOR_PROJECT_ROOT="${PROJECT_ROOT}" \
    -e WILOR_VENV_DIR="${VENV_DIR}" \
    -e WILOR_PYPI_MIRROR="${PYPI_MIRROR}" \
    -e WILOR_PROXY_ENV="${PROXY_ENV}" \
    -e WILOR_APT_PACKAGES="${WILOR_APT_PACKAGES:-${APT_PACKAGES[*]}}" \
    -e WILOR_SHARED_APT_CACHE="${SHARED_APT_CACHE}" \
    -e WILOR_INSTALL_APT_LIBS="${WILOR_INSTALL_APT_LIBS:-1}" \
    -e UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
    -e WILOR_HOST_UID="$(id -u)" \
    -e WILOR_HOST_GID="$(id -g)" \
    "${IMAGE_TAG}" \
    bash -lc "bash scripts/bootstrap_musa_uv.sh"
}

find_uv() {
  if [ -x "${VENV_DIR}/bin/python" ] && "${VENV_DIR}/bin/python" -m uv --version >/dev/null 2>&1; then
    UV_CMD=("${VENV_DIR}/bin/python" -m uv)
  elif command -v uv >/dev/null 2>&1; then
    UV_CMD=(uv)
  elif [ -x "${HOME:-/root}/.local/bin/uv" ]; then
    UV_CMD=("${HOME:-/root}/.local/bin/uv")
  elif python3 -m uv --version >/dev/null 2>&1; then
    UV_CMD=(python3 -m uv)
  else
    return 1
  fi
}

acquire_lock() {
  mkdir -p "$(dirname "${LOCK_FILE}")"
  exec 9>"${LOCK_FILE}"
  flock 9
}

venv_uses_system_site_packages() {
  [ -f "${VENV_DIR}/pyvenv.cfg" ] \
    && grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true$' "${VENV_DIR}/pyvenv.cfg"
}

validate_runtime() {
  "${VENV_DIR}/bin/python" - <<'PY'
import importlib
import importlib.metadata as metadata
import sys

expected = {
    "torch": "2.7.1",
    "torch_musa": "2.7.1+9f1bb31",
    "torchvision": "0.22.1+32091f2",
}

errors = []
for module_name, version in expected.items():
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        errors.append(f"{module_name}: import failed: {exc!r}")
        continue
    actual = getattr(module, "__version__", None)
    if actual != version:
        errors.append(f"{module_name}: expected {version}, got {actual}")

try:
    import torch
    if not torch.musa.is_available():
        errors.append("torch.musa.is_available() is false")
except Exception as exc:
    errors.append(f"torch.musa check failed: {exc!r}")

for dist_name in ("torch", "torchvision"):
    try:
        metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        errors.append(f"{dist_name}: distribution metadata is not visible")

if errors:
    print("WiLoR MUSA runtime validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

ensure_system_libs() {
  [ "${WILOR_INSTALL_APT_LIBS:-1}" = "1" ] || return 0

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is required to install EGL/GLVND runtime libraries." >&2
    return 1
  fi

  bash "${SCRIPT_DIR}/prepare_musa_syslibs.sh"
  if [ -n "${WILOR_HOST_UID:-}" ] && [ -n "${WILOR_HOST_GID:-}" ]; then
    chown -R "${WILOR_HOST_UID}:${WILOR_HOST_GID}" "${SHARED_APT_CACHE}" 2>/dev/null || true
  fi

  # shellcheck disable=SC1091
  source "${SHARED_APT_CACHE}/env.sh"
}

validate_no_managed_stack_in_venv() {
  "${VENV_DIR}/bin/python" - <<'PY'
import importlib.metadata as metadata
import sys
import sysconfig

site = sysconfig.get_paths()["purelib"]
blocked_names = {
    "torch",
    "torchvision",
    "torch-musa",
    "torchaudio",
    "triton",
    "opencv-python",
}
blocked = []
for dist in metadata.distributions(path=[site]):
    name = dist.metadata.get("Name", "").lower().replace("_", "-")
    if name in blocked_names or name.startswith("nvidia-") or name.startswith("cuda-"):
        blocked.append(name)

if blocked:
    print("Blocked packages were installed into .venv:", file=sys.stderr)
    for name in sorted(blocked):
        print(f"  - {name}", file=sys.stderr)
    raise SystemExit(1)
PY
}

venv_has_forbidden_installs() {
  [ -x "${VENV_DIR}/bin/python" ] || return 1
  validate_no_managed_stack_in_venv >/dev/null 2>&1
}

env_stamp() {
  sha256sum \
    "${PROJECT_ROOT}/pyproject.toml" \
    "${PROJECT_ROOT}/constraints-musa.txt" \
    "${PROJECT_ROOT}/overrides-musa.txt" \
    "${PROJECT_ROOT}/excludes-musa.txt" \
    "${PROJECT_ROOT}/scripts/bootstrap_musa_uv.sh" \
    "${PROJECT_ROOT}/scripts/prepare_musa_syslibs.sh" \
    | sha256sum \
    | awk '{print $1}'
}

validate_app_imports() {
  "${VENV_DIR}/bin/python" - <<'PY'
import importlib
import sys

modules = [
    "cv2",
    "pyrender",
    "pytorch_lightning",
    "skimage",
    "smplx",
    "yacs",
    "timm",
    "einops",
    "xtcocotools",
    "hydra",
    "pyrootutils",
    "rich",
    "webdataset",
    "gradio",
    "ultralytics",
    "wilor",
]

errors = []
for module_name in modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        errors.append(f"{module_name}: {exc!r}")

try:
    from OpenGL.osmesa import OSMesaCreateContextAttribs  # noqa: F401
except Exception as exc:
    errors.append(f"OpenGL.osmesa.OSMesaCreateContextAttribs: {exc!r}")

if errors:
    print("WiLoR application import validation failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    raise SystemExit(1)
PY
}

if [ "${WILOR_BOOTSTRAP_IN_CONTAINER:-0}" != "1" ] && ! in_mtwan_image; then
  run_in_container
  exit $?
fi

cd "${PROJECT_ROOT}"

ensure_system_libs

if [ -f "${PROXY_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${PROXY_ENV}"
fi

export PATH="${HOME:-/root}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

acquire_lock

if ! find_uv; then
  env PYTHONNOUSERSITE= python3 -m pip install --user -i "${PYPI_MIRROR}" uv
  find_uv
fi

if [ ! -x "${VENV_DIR}/bin/python" ] || ! venv_uses_system_site_packages || ! venv_has_forbidden_installs; then
  rm -rf "${VENV_DIR}"
  if [ "${UV_CMD[0]:-}" = "${VENV_DIR}/bin/python" ]; then
    UV_CMD=()
    if ! find_uv; then
      env PYTHONNOUSERSITE= python3 -m pip install --user -i "${PYPI_MIRROR}" uv
      find_uv
    fi
  fi
  "${UV_CMD[@]}" venv --python /usr/bin/python3 --system-site-packages "${VENV_DIR}"
fi

"${UV_CMD[@]}" pip install \
  --python "${VENV_DIR}/bin/python" \
  uv \
  --index-url "${PYPI_MIRROR}" \
  --index-strategy unsafe-best-match

UV_CMD=("${VENV_DIR}/bin/python" -m uv)

validate_runtime

CURRENT_STAMP="$(env_stamp)"
INSTALLED_STAMP=""
if [ -f "${STAMP_FILE}" ]; then
  INSTALLED_STAMP="$(cat "${STAMP_FILE}")"
fi

if [ "${CURRENT_STAMP}" != "${INSTALLED_STAMP}" ] || ! validate_no_managed_stack_in_venv || ! validate_app_imports; then
  "${UV_CMD[@]}" pip install \
    --python "${VENV_DIR}/bin/python" \
    -e "${PROJECT_ROOT}" \
    --constraints "${PROJECT_ROOT}/constraints-musa.txt" \
    --overrides "${PROJECT_ROOT}/overrides-musa.txt" \
    --excludes "${PROJECT_ROOT}/excludes-musa.txt" \
    --index-url "${PYPI_MIRROR}" \
    --index-strategy unsafe-best-match
  printf '%s\n' "${CURRENT_STAMP}" > "${STAMP_FILE}"
fi

validate_runtime
validate_no_managed_stack_in_venv
validate_app_imports

if [ -n "${WILOR_HOST_UID:-}" ] && [ -n "${WILOR_HOST_GID:-}" ]; then
  chown -R "${WILOR_HOST_UID}:${WILOR_HOST_GID}" "${VENV_DIR}" "${LOCK_FILE}" 2>/dev/null || true
  if [ -d "${PROJECT_ROOT}/wilor.egg-info" ]; then
    chown -R "${WILOR_HOST_UID}:${WILOR_HOST_GID}" "${PROJECT_ROOT}/wilor.egg-info" 2>/dev/null || true
  fi
fi

echo "WiLoR MUSA uv environment ready: ${VENV_DIR}"
