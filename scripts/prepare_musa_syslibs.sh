#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${WILOR_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SHARED_CACHE_ROOT="${WILOR_SHARED_APT_CACHE:-/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd}"
PROXY_ENV="${WILOR_PROXY_ENV:-/datapool/.config/mihomo/proxy-env.sh}"
APT_STATE_DIR="${SHARED_CACHE_ROOT}/apt-state"
DEB_DIR="${SHARED_CACHE_ROOT}/debs"
ROOT_DIR="${SHARED_CACHE_ROOT}/root"
SOURCES_FILE="${SHARED_CACHE_ROOT}/sources.list"
PACKAGES=(${WILOR_APT_PACKAGES:-libglvnd0 libegl1 libgl1 libglx0 libopengl0 libosmesa6})
MIRRORS=(
  "https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
  "https://mirrors.aliyun.com/ubuntu"
  "https://mirrors.ustc.edu.cn/ubuntu"
)

mkdir -p "${APT_STATE_DIR}/lists/partial" "${SHARED_CACHE_ROOT}/cache/archives/partial" "${DEB_DIR}" "${ROOT_DIR}"

apt_with_source() {
  apt-get \
    -o "Dir::Etc::sourcelist=${SOURCES_FILE}" \
    -o "Dir::Etc::sourceparts=-" \
    -o "Dir::State::Lists=${APT_STATE_DIR}/lists" \
    -o "Dir::Cache=${SHARED_CACHE_ROOT}/cache" \
    -o "Dir::Cache::archives=${DEB_DIR}" \
    -o "APT::Get::List-Cleanup=0" \
    "$@"
}

write_sources() {
  local mirror="$1"
  cat > "${SOURCES_FILE}" <<EOF
deb ${mirror} jammy main restricted universe multiverse
deb ${mirror} jammy-updates main restricted universe multiverse
deb ${mirror} jammy-backports main restricted universe multiverse
deb ${mirror} jammy-security main restricted universe multiverse
EOF
}

cache_has_packages() {
  local package
  for package in "${PACKAGES[@]}"; do
    if ! find "${DEB_DIR}" -maxdepth 1 -type f -name "${package}_*.deb" | grep -q .; then
      return 1
    fi
  done
}

root_has_libraries() {
  [ -e "${ROOT_DIR}/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0" ] \
    && [ -e "${ROOT_DIR}/usr/lib/x86_64-linux-gnu/libEGL.so.1" ] \
    && [ -e "${ROOT_DIR}/usr/lib/x86_64-linux-gnu/libGL.so.1" ] \
    && [ -e "${ROOT_DIR}/usr/lib/x86_64-linux-gnu/libOSMesa.so.8" ]
}

write_env() {
  cat > "${SHARED_CACHE_ROOT}/env.sh" <<EOF
export WILOR_SYSROOT="${ROOT_DIR}"
export LD_LIBRARY_PATH="${ROOT_DIR}/usr/lib/x86_64-linux-gnu:${ROOT_DIR}/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="${ROOT_DIR}/usr/share/glvnd/egl_vendor.d:\${__EGL_VENDOR_LIBRARY_DIRS:-}"
export LIBGL_DRIVERS_PATH="${ROOT_DIR}/usr/lib/x86_64-linux-gnu/dri:\${LIBGL_DRIVERS_PATH:-}"
export PYOPENGL_PLATFORM="\${PYOPENGL_PLATFORM:-osmesa}"
export EGL_PLATFORM="\${EGL_PLATFORM:-surfaceless}"
export MESA_LOADER_DRIVER_OVERRIDE="\${MESA_LOADER_DRIVER_OVERRIDE:-swrast}"
export LIBGL_ALWAYS_SOFTWARE="\${LIBGL_ALWAYS_SOFTWARE:-1}"
EOF
}

extract_packages() {
  local deb
  local libdir
  rm -rf "${ROOT_DIR}"
  mkdir -p "${ROOT_DIR}"
  for deb in "${DEB_DIR}"/*.deb; do
    dpkg-deb -x "${deb}" "${ROOT_DIR}"
  done
  libdir="${ROOT_DIR}/usr/lib/x86_64-linux-gnu"
  [ -e "${libdir}/libEGL.so.1" ] && ln -sfn libEGL.so.1 "${libdir}/EGL"
  [ -e "${libdir}/libGL.so.1" ] && ln -sfn libGL.so.1 "${libdir}/GL"
  [ -e "${libdir}/libGLX.so.0" ] && ln -sfn libGLX.so.0 "${libdir}/GLX"
  [ -e "${libdir}/libOpenGL.so.0" ] && ln -sfn libOpenGL.so.0 "${libdir}/OpenGL"
  [ -e "${libdir}/libOSMesa.so.8" ] && ln -sfn libOSMesa.so.8 "${libdir}/OSMesa"
  write_env
}

download_with_current_env() {
  local mirror="$1"
  write_sources "${mirror}"
  apt_with_source update
  apt_with_source install \
    --download-only \
    --reinstall \
    -y \
    --no-install-recommends \
    "${PACKAGES[@]}"
}

download_without_proxy() {
  local mirror
  for mirror in "${MIRRORS[@]}"; do
    echo "Trying direct domestic Ubuntu mirror: ${mirror}"
    if (
      unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
      download_with_current_env "${mirror}"
    ); then
      echo "${mirror}" > "${SHARED_CACHE_ROOT}/mirror-used.txt"
      return 0
    fi
  done
  return 1
}

download_with_proxy() {
  [ -f "${PROXY_ENV}" ] || return 1
  # shellcheck disable=SC1090
  source "${PROXY_ENV}"

  local mirror
  for mirror in "${MIRRORS[@]}"; do
    echo "Trying proxied domestic Ubuntu mirror: ${mirror}"
    if download_with_current_env "${mirror}"; then
      echo "${mirror} via proxy" > "${SHARED_CACHE_ROOT}/mirror-used.txt"
      return 0
    fi
  done

  echo "Trying proxied default Ubuntu archive."
  if download_with_current_env "http://archive.ubuntu.com/ubuntu"; then
    echo "http://archive.ubuntu.com/ubuntu via proxy" > "${SHARED_CACHE_ROOT}/mirror-used.txt"
    return 0
  fi

  return 1
}

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required to prepare MUSA system libraries." >&2
  exit 1
fi

if cache_has_packages; then
  if ! root_has_libraries; then
    extract_packages
  fi
  if ! grep -q 'MESA_LOADER_DRIVER_OVERRIDE' "${SHARED_CACHE_ROOT}/env.sh" 2>/dev/null; then
    write_env
  fi
  echo "Using cached MUSA system libraries: ${DEB_DIR}"
  exit 0
fi

rm -f "${DEB_DIR}"/*.deb

if ! download_without_proxy; then
  echo "Direct domestic mirrors failed; retrying with proxy." >&2
  download_with_proxy
fi

if ! cache_has_packages; then
  echo "MUSA system library cache is incomplete: ${DEB_DIR}" >&2
  exit 1
fi

extract_packages

cat > "${SHARED_CACHE_ROOT}/packages.txt" <<EOF
${PACKAGES[*]}
EOF

echo "MUSA system libraries cached in ${DEB_DIR}"
