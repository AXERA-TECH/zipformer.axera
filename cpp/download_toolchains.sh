#!/usr/bin/env bash
# 下载交叉编译器与 AXERA BSP SDK（默认放到 cpp/toolchains/，不随 git 提交）。
#
# 用法:
#   bash cpp/download_toolchains.sh            # 全部 (gcc + 650 + 630C)
#   bash cpp/download_toolchains.sh 650        # 仅 AX650 BSP
#   bash cpp/download_toolchains.sh 630C       # 仅 AX630C BSP
#
# 若已把工具链放在其他目录，可导出环境变量后直接编译：
#   export TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu
#   export BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out
#   bash cpp/build_ax650.sh
set -euo pipefail

CPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAINS_DIR="${CPP_DIR}/toolchains"
PLATFORM="${1:-all}"

GCC_URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
GCC_DIR="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

download_gcc() {
  if [[ -x "${TOOLCHAINS_DIR}/${GCC_DIR}/bin/aarch64-none-linux-gnu-g++" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/${GCC_DIR}"
    return
  fi
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Downloading ${GCC_URL}"
  wget -q --show-progress "${GCC_URL}" -O gcc.tar.xz
  tar -xf gcc.tar.xz
  rm -f gcc.tar.xz
  echo "Done: ${TOOLCHAINS_DIR}/${GCC_DIR}"
}

download_sdk() {
  local directory="$1"
  local url="$2"
  if [[ -d "${TOOLCHAINS_DIR}/${directory}" ]]; then
    echo "Found ${TOOLCHAINS_DIR}/${directory}"
    return
  fi
  mkdir -p "${TOOLCHAINS_DIR}"
  cd "${TOOLCHAINS_DIR}"
  echo "Cloning ${url}"
  git clone "${url}" --depth=1
  echo "Done: ${TOOLCHAINS_DIR}/${directory}"
}

case "${PLATFORM}" in
  all | 650 | 630C)
    download_gcc
    ;;
esac

case "${PLATFORM}" in
  all | 650)
    download_sdk ax650n_bsp_sdk https://github.com/AXERA-TECH/ax650n_bsp_sdk.git
    ;;
esac

case "${PLATFORM}" in
  all | 630C)
    download_sdk ax620e_bsp_sdk https://github.com/AXERA-TECH/ax620e_bsp_sdk.git
    ;;
  *)
    echo "Usage: bash cpp/download_toolchains.sh [all|650|630C]" >&2
    exit 2
    ;;
esac
