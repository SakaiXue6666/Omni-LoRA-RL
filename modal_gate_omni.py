"""Phase 0.1b —— 在新版 Relax 官方镜像上，注册 Qwen3-Omni bridge 后重验识别。

为什么需要这一步：
    modal_env_gate_v2.py --mode gate 已跑通，4 个判据里 3 个 PASS
    （megatron.core 0.18.0 + megatron.bridge 0.5.0 共存、adapter 导出 API 齐全、
      bridge PEFT LoRA 可导入），只有 [omni] 报
          Model architecture 'Qwen3OmniMoeForConditionalGeneration' is not yet supported

    但这是探测脚本的问题，不是环境的问题。Qwen3-Omni 的 bridge 不在 megatron-bridge
    的内置支持表里，而是 Relax 在 relax/models/__init__.py 里通过
    @MegatronModelBridge.register_bridge 注册的。mig_00_env.py 沿用 2026-07 的逻辑，
    只 import megatron.bridge、从未 import relax，所以注册没发生。

    本脚本把 Relax 源码放进镜像并 import relax.models 之后再验一次。

Relax 源码的引入方式：
    用 git clone + PYTHONPATH，而不是 pip install。原因是官方镜像构建于 2026-07-23，
    而 main 已经往前走了两周；pip install 会跑 setup.py 并可能触发依赖解析，
    把镜像里钉好的 torch/flashinfer 版本搅乱——那正是 2026-07 迁移失败的原因。
    纯 PYTHONPATH 注入不动任何已装依赖。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_gate_omni.py
    modal run modal_gate_omni.py --relax-ref <commit>   # 钉具体 commit
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_01_omni_bridge.py"
REMOTE_SCRIPT = "/root/mig_01_omni_bridge.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

# 与 modal_env_gate_v2.py 同一个 digest：ghcr.io/redai-infra/relaxrl:latest
# == dev-20260723-8cc1e8fd，解析于 2026-08-10。
RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

RELAX_REPO = "https://github.com/redai-infra/Relax.git"
RELAX_REF = os.environ.get("RELAX_V2_REF", "main")
RELAX_SRC = "/root/RelaxSrc"

gate_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none {RELAX_REPO} {RELAX_SRC}",
        f"cd {RELAX_SRC} && git checkout {RELAX_REF} && git log -1 --format='RELAX_HEAD %H %cI %s'",
    )
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .env(
        {
            "OMNI_CKPT": OMNI_CKPT,
            # 镜像原有 PYTHONPATH 是 /root/Megatron-LM/:/pkg/:/root/，必须保留，
            # megatron.core 就是从 /root/Megatron-LM 来的。
            "PYTHONPATH": f"{RELAX_SRC}:/root/Megatron-LM/:/pkg/:/root/",
        }
    )
)

app = modal.App("relax-omni-gate-0-1b")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=gate_image, gpu="T4", timeout=30 * 60, volumes={MODEL_MOUNT: model_volume})
def gate_omni() -> int:
    import subprocess
    import sys

    print("========== Relax 源码版本 ==========", flush=True)
    subprocess.run(
        ["git", "log", "-1", "--format=commit %H%ndate   %cI%nsubject %s"],
        cwd=RELAX_SRC,
        check=False,
    )
    print("PYTHONPATH =", os.environ.get("PYTHONPATH"), flush=True)

    print("\n========== 运行 mig_01_omni_bridge.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_01_omni_bridge.py returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = gate_omni.remote()
    if rc != 0:
        raise SystemExit(f"Phase 0.1b 未通过 (returncode={rc})")
    print("\n[GATE OK] Qwen3-Omni 可被识别，Phase 0.1 全部判据通过。", flush=True)
