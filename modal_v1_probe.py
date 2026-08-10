"""定位 v1 真实使用的 slime 镜像 —— 走 Modal 镜像缓存，而不是猜 tag。

为什么不能靠猜 tag：
    v1 的所有 Modal 脚本都写 `slimerl/slime:latest`（可变 tag）。按日期猜的两个候选
    实测都不匹配 MIGRATION_PLAN.md 记录的 v1 运行时版本：

        MIGRATION_PLAN.md 记录（2026-07-07 第 2 轮）：transformers 4.57.1、flashinfer 0.6.3
        nightly-dev-20260707a 实测：      transformers 5.8.1  flashinfer-python 0.6.12
        nightly-dev-20260722a 实测：      transformers 5.12.1 flashinfer-python 0.6.12

    而且这两个 nightly 都已自带 megatron-bridge 0.5.0 —— 若 v1 用的是它们，
    七月"pip 装 bridge 0.5.0 → megatron.core 太旧 / flashinfer 冲突"那两轮踩坑
    根本不会以那种形式发生。redai bridge 那行用的是 --no-deps --force-reinstall，
    也不会把 transformers 降到 4.57.1。结论：v1 的 base 比这两个都老。

本脚本的思路：
    Modal 按"镜像定义"缓存镜像，`from_registry("slimerl/slime:latest")` 只要定义没变
    就复用当初构建的那一份，不会每次重新解析 tag。v1 用的正是这一行定义，
    所以这里原样复刻它、只在容器里 dump 版本，读到的就是 v1 的真实环境。

判读：
    - 若 transformers ≈ 4.57.1 且 flashinfer ≈ 0.6.3  -> 命中 v1 缓存，据此反查 nightly tag 并钉 digest。
    - 若版本是 5.1x / 0.6.12                          -> 缓存已失效，Modal 重新解析了 latest，
                                                        改用 dist-packages 的构建时间反推 tag。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_v1_probe.py
"""

from __future__ import annotations

import modal

# 与 modal_verify_lora.py / modal_relax_smoke.py 第一行镜像定义完全一致，
# 改动任何字符都可能错过缓存。
V1_BASE = "slimerl/slime:latest"

PROBE_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "megatron-core",
    "megatron-bridge",
    "transformer-engine",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "sglang",
    "ray",
    "numpy",
)

_PROBE = r"""
import os, sys, time
from importlib.metadata import version, PackageNotFoundError

print("PYTHON:", sys.version.replace("\n", " "), flush=True)
print("========== 关键包版本 ==========", flush=True)
for name in {wanted!r}:
    try:
        print(f"  {{name}}: {{version(name)}}", flush=True)
    except PackageNotFoundError:
        print(f"  {{name}}: <未安装>", flush=True)

# 镜像构建时间的代理指标：site-packages 里几个包的 mtime。
# slime 的 nightly tag 按日期发布，用它可以反推是哪一天那份。
print("========== 构建时间线索（mtime）==========", flush=True)
import glob
for pat in ("/usr/local/lib/python3*/dist-packages/torch/version.py",
            "/usr/local/lib/python3*/dist-packages/transformers/__init__.py",
            "/usr/local/lib/python3*/site-packages/torch/version.py"):
    for p in glob.glob(pat):
        print(f"  {{p}}: {{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(os.path.getmtime(p)))}} UTC", flush=True)
""".format(wanted=list(PROBE_PACKAGES))

image = modal.Image.from_registry(V1_BASE, add_python=None)

app = modal.App("v1-image-probe")


@app.function(image=image, timeout=15 * 60)
def probe() -> None:
    import subprocess
    import sys

    print(f"########## {V1_BASE}（Modal 缓存视角）##########", flush=True)
    subprocess.run([sys.executable, "-c", _PROBE], check=False)


@app.local_entrypoint()
def main() -> None:
    probe.remote()
