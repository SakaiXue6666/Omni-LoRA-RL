"""Phase 0 —— 在 Modal 上验证迁移到上游 Megatron-Bridge 0.5.0 是否可行。

对应 MIGRATION_PLAN.md 的 Phase 0.1（环境/API 兼容闸门）。仿 modal_verify_lora.py。

策略：
  - 基础镜像沿用 slime 预构建镜像（自带 Megatron-LM + TE + megatron.core）。
  - 历史（第 1 轮，2026-07-07）：`--no-deps --force-reinstall` 只换 bridge 代码、
    保留 slime 的 megatron.core/TE —— 结果 import 即炸：
    `cannot import name 'safe_get_world_size' from 'megatron.core._rank_utils'`
    （slime 自带 core 太旧，bridge 0.5.0 要求更新的 core API）。结论：最小改动不可行。
  - 当前（第 2 轮）：**去掉 --no-deps 且不 force-reinstall**，让 pip 按依赖把
    megatron-core 升到 bridge 0.5.0 要求的版本，同时尽量保留 slime 已编译的 torch/TE。
    关键观察点在 mig_00_env.py 打印的 `megatron.core: <版本> @ <路径>`：
      · 若路径指向 dist-packages 的新版且 import 通过 -> 可迁移（core 被成功升级）；
      · 若仍指向 /root/Megatron-LM 旧版（被 .pth/PYTHONPATH 遮蔽）-> 需换基础镜像。
  - 挂载已缓存的 Qwen3-Omni 权重卷（qwen3-omni-weights，由 modal_run.py 下好），
    供 from_hf_pretrained 识别。
  - 最便宜的 T4，几分钟跑完。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_migrate.py

可调（环境变量）：
    MIG_BASE_IMAGE       基础镜像（默认 slimerl/slime:latest）
    MIG_BRIDGE_INSTALL   bridge 安装命令（默认见下；想只换 bridge 加回 --no-deps；
                         想装 git ref 改成 pip install 'megatron-bridge @ git+https://...'）
"""

from __future__ import annotations

import os
import pathlib

import modal

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_00_env.py"
REMOTE_SCRIPT = "/root/mig_00_env.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"                 # 卷挂载点（与 modal_run.py 一致）
OMNI_CKPT = "/models/qwen3-omni"        # 权重目录（卷内）

BASE_IMAGE = os.environ.get("MIG_BASE_IMAGE", "slimerl/slime:latest")
BRIDGE_INSTALL = os.environ.get(
    "MIG_BRIDGE_INSTALL",
    # 第 2 轮：带依赖安装，让 pip 按需升级 megatron-core（不 force-reinstall，
    # 尽量保留 slime 已编译的 torch/TE）。第 1 轮的 --no-deps 已证实不可行。
    "pip install --no-cache-dir megatron-bridge==0.5.0",
)

# ---------------------------------------------------------------------------
# 镜像：slime 基础镜像 + 上游 bridge 0.5.0。
# `|| true` 让 build 不因装失败而中断，好让 mig_00_env.py 在容器里打印精确的
# import 错误用于诊断（装失败本身就是 Phase 0.1 要暴露的信息）。
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .run_commands(f"{BRIDGE_INSTALL} || true")
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .env({"OMNI_CKPT": OMNI_CKPT})
)

app = modal.App("relax-omni-migrate")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=15 * 60, volumes={MODEL_MOUNT: model_volume})
def env_gate() -> None:
    """Phase 0.1：上游 bridge 0.5.0 环境/API 兼容闸门。"""
    import subprocess
    import sys

    print("========== pip show megatron-bridge / megatron-core ==========", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "show", "megatron-bridge", "megatron-core"],
        check=False,
    )

    print("\n========== 运行 mig_00_env.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_00_env.py returncode={proc.returncode}", flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


@app.local_entrypoint()
def main() -> None:
    env_gate.remote()
