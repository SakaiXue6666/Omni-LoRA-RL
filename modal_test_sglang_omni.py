"""在 Modal 上跑 sglang-omni 的 LoRA 接线「CPU 单测」（最小成本，无 GPU）。

目的：验证给 sglang-omni 加的 LoRA 热加载接线（admin dispatch / ModelWorker
委托 / OmniScheduler enable_lora helper / thinker should_apply_lora gate）在真实
sglang 0.5.12 环境下能 import 且逻辑正确。只跑纯 CPU 单测，不加载模型、不起 GPU。

对应单测：sglang-omni/tests/unit_test/scheduling/test_lora_admin.py

【镜像策略】复用 modal_relax_smoke 的 slime 预构建镜像（已含 torch），把【本地
这份】sglang（0.5.12 fork，含 Block1/2 LoRA 改动）+ sglang-omni 源码一起用
PYTHONPATH 注入，再 pip 装 pytest。sglang-omni 依赖 pip 装的 sglang，这里用
PYTHONPATH 覆盖成本地 fork（版本已对齐 0.5.12.post1）。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_test_sglang_omni.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
SGLANG_LOCAL = HERE / "sglang" / "python"        # 本地 sglang 0.5.12 fork（Block1/2）
SGLANG_OMNI_LOCAL = HERE / "sglang-omni"         # 本地 sglang-omni（含 LoRA 接线）

SGLANG_REMOTE = "/root/sglang_src/python"        # PYTHONPATH 最前，覆盖镜像自带 sglang
SGLANG_OMNI_REMOTE = "/root/sglang-omni"         # 仓库根（pyproject pythonpath=["."]）

# sglang 最前 -> sglang-omni 仓库根
PYTHONPATH = f"{SGLANG_REMOTE}:{SGLANG_OMNI_REMOTE}"

BASE_IMAGE = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .run_commands(
        "pip install --no-cache-dir pytest pytest-asyncio || true",
        # 轻量依赖：让 CLI / coordinator / client 接线单测也能真正跑（缺则 skip）
        "pip install --no-cache-dir typer pyzmq msgpack pydantic pyyaml xxhash "
        "httpx fastapi uvicorn pybase64 || true",
    )
    .add_local_dir(
        SGLANG_LOCAL.as_posix(),
        SGLANG_REMOTE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        SGLANG_OMNI_LOCAL.as_posix(),
        SGLANG_OMNI_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/node_modules"],
    )
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App("sglang-omni-lora-unittest")


@app.function(image=image, cpu=2.0, timeout=900)
def run_unit_tests() -> None:
    import subprocess
    import sys

    # 探针 1：本地 sglang fork 能否 import（sgl-kernel/flashinfer 二进制对得上）
    print("=" * 70)
    print("[probe] import sglang ...")
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sglang; "
            "print('sglang from', sglang.__file__); "
            "from sglang.srt.lora.lora_manager import LoRAManager; "
            "print('LoRAManager import OK')",
        ],
        cwd=SGLANG_OMNI_REMOTE,
    )
    if r.returncode != 0:
        raise SystemExit("[probe] import sglang FAILED — 后面单测无意义，先修镜像")

    # 探针 2：能否 import 我们改过的两个 omni 模块（真正的接线是否 import 得动）
    print("=" * 70)
    print("[probe] import sglang_omni scheduler + model_worker ...")
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sglang_omni.scheduling.omni_scheduler import OmniScheduler; "
            "from sglang_omni.model_runner.model_worker import ModelWorker; "
            "print('omni import OK')",
        ],
        cwd=SGLANG_OMNI_REMOTE,
    )
    if r.returncode != 0:
        raise SystemExit("[probe] import sglang_omni FAILED")

    # 正式单测
    print("=" * 70)
    print("[pytest] running LoRA wiring unit tests ...")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit_test/scheduling/test_lora_admin.py",
            "-v",
            "-p",
            "no:cacheprovider",
        ],
        cwd=SGLANG_OMNI_REMOTE,
    )
    if r.returncode != 0:
        raise SystemExit(f"[pytest] FAILED with code {r.returncode}")
    print("=" * 70)
    print("[pytest] ALL PASSED")


@app.local_entrypoint()
def main() -> None:
    run_unit_tests.remote()
