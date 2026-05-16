# WiLoR MUSA uv 使用说明

本文记录 `/datapool/husicheng/WiLoR` 在 Moore Threads MUSA 机器上的 uv 环境使用方式。

## 环境边界

- 默认镜像：`registry.mthreads.com/mcctest/ai/mtwan:4.3.3-pt2.7-v0.2`
- 共享镜像 loader：`/datapool/shared_images/mtwan_4.3.3-pt2.7-v0.2/load_mtwan.sh`
- 项目 venv：`/datapool/husicheng/WiLoR/.venv`
- 系统库缓存：`/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd`
- PyPI 镜像：`https://pypi.tuna.tsinghua.edu.cn/simple`

`.venv` 使用 `--system-site-packages` 创建，复用 mtwan 镜像内的 MUSA PyTorch：

```text
torch==2.7.1
torch_musa==2.7.1+9f1bb31
torchvision==0.22.1+32091f2
```

不要在项目 `.venv` 内安装 PyPI/CUDA 版 `torch`、`torchvision`、`torch_musa`、`triton` 或 `nvidia-*`/`cuda-*` 包。`constraints-musa.txt`、`excludes-musa.txt` 和 bootstrap 校验会阻止这些包进入 `.venv`。

## 初始化

从宿主机运行：

```bash
cd /datapool/husicheng/WiLoR
bash scripts/bootstrap_musa_uv.sh
```

脚本会自动进入默认 mtwan 容器。若本机没有镜像，会先从共享目录加载默认镜像。

系统层渲染库不会安装进容器系统目录。`scripts/prepare_musa_syslibs.sh` 会先直连国内 Ubuntu 22.04 镜像下载 `.deb` 到 `/datapool/shared_apt/ubuntu-jammy/wilor-egl-glvnd`，并解包成共享 sysroot；只有国内镜像直连失败时才加载 `/datapool/.config/mihomo/proxy-env.sh` 重试。运行时通过 `LD_LIBRARY_PATH`、`PYOPENGL_PLATFORM=osmesa`、`LIBGL_DRIVERS_PATH` 等环境变量使用这份共享 sysroot。

## 运行 Demo

所有运行命令建议通过 `scripts/run_musa.sh` 进入同一 mtwan 容器：

```bash
cd /datapool/husicheng/WiLoR
bash scripts/run_musa.sh .venv/bin/python demo.py \
  --img_folder demo_img \
  --out_folder demo_out \
  --save_mesh
```

快速模式：

```bash
bash scripts/run_musa.sh .venv/bin/python demo.py \
  --img_folder demo_img \
  --out_folder demo_out_fast \
  --save_mesh \
  --fast
```

Gradio：

```bash
bash scripts/run_musa.sh .venv/bin/python gradio_demo.py
```

## 验证

检查 MUSA PyTorch 和 `.venv` 排除项：

```bash
bash scripts/run_musa.sh .venv/bin/python - <<'PY'
import importlib.metadata as metadata
import sysconfig
from pathlib import Path

import torch
import torch_musa
import torchvision
from wilor.utils.device import get_torch_device

site = Path(sysconfig.get_paths()["purelib"])
blocked = []
for dist in metadata.distributions(path=[str(site)]):
    name = dist.metadata.get("Name", "").lower().replace("_", "-")
    if name in {"torch", "torchvision", "torch-musa", "torchaudio", "triton", "opencv-python"}:
        blocked.append(name)
    if name.startswith("nvidia-") or name.startswith("cuda-"):
        blocked.append(name)

print("device", get_torch_device())
print("torch", torch.__version__)
print("torch_musa", torch_musa.__version__)
print("torchvision", torchvision.__version__)
print("torch.version.musa", getattr(torch.version, "musa", None))
print("musa available", torch.musa.is_available())
print("blocked in venv", sorted(blocked))
PY
```

期望输出包含：

```text
device musa
torch 2.7.1
torch_musa 2.7.1+9f1bb31
torchvision 0.22.1+32091f2
musa available True
blocked in venv []
```

## 可配置项

- `WILOR_IMAGE`：覆盖 mtwan 镜像。
- `WILOR_VENV_DIR`：覆盖项目 venv 路径。
- `WILOR_PYPI_MIRROR`：覆盖 Python 包镜像。
- `WILOR_SHARED_APT_CACHE`：覆盖系统库共享缓存目录。
- `WILOR_APT_PACKAGES`：覆盖需要缓存的 Ubuntu 包列表。
- `WILOR_PROXY_ENV`：覆盖代理环境脚本路径。
- `WILOR_INSTALL_APT_LIBS=0`：跳过共享系统库准备；仅在外部已提供可用 OSMesa/EGL 环境时使用。

## 已验证

已在当前机器上完成：

- `bash scripts/bootstrap_musa_uv.sh`
- MUSA runtime import 校验
- `.venv` 中 torch/CUDA 排除项扫描
- 单图 demo smoke：`demo_img/test2.png` 输出渲染 jpg 和 mesh obj

最近一次 smoke 输出目录：

```text
/datapool/husicheng/tmp/260516-1925/wilor_demo/out
```
