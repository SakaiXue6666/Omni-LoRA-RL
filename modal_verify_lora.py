"""在 Modal 上跑 verify_lora_attach.py（机制验证，成本最小化）。

省钱策略：
  - 直接拉 slime 预构建镜像 `slimerl/slime:latest`（已装 Megatron + Megatron-Bridge
    + TransformerEngine），不从零编译（否则编 flash-attn/TE/apex 要几小时 + 大量费用）。
  - 用最便宜的 GPU(T4)；脚本只跑几秒。
  - 短超时，跑完即停。
  - 镜像拉取发生在 image build 阶段（不占 GPU 计费），函数运行只占 T4 几十秒。

用法（d:\\Li_Lab\\RL 下）：
    chcp 65001; $env:PYTHONUTF8=1
    modal run modal_verify_lora.py
"""

from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "verify_lora_attach.py"
REMOTE_SCRIPT = "/root/verify_lora_attach.py"
LOCAL_TP_SCRIPT = HERE / "verify_lora_tp.py"
REMOTE_TP_SCRIPT = "/root/verify_lora_tp.py"

image = (
    modal.Image.from_registry("slimerl/slime:latest", add_python=None)
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .add_local_file(LOCAL_TP_SCRIPT.as_posix(), REMOTE_TP_SCRIPT, copy=True)
)

app = modal.App("verify-lora-attach")


@app.function(image=image, gpu="T4", timeout=10 * 60)
def run_verify() -> None:
    import subprocess
    import sys

    print("========== 环境自检 ==========")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, megatron.core; "
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); "
            "import megatron.bridge as mb; print('megatron.bridge OK', getattr(mb,'__file__',''))",
        ],
        check=False,
    )

    print("\n========== 运行验证脚本 ==========")
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] verify_lora_attach.py returncode={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


@app.function(image=image, gpu="T4:2", timeout=15 * 60)
def run_verify_tp() -> None:
    """验证 M:TP=2 下 adapter 的 TP 切分复原(2×T4,torchrun 起 2 进程)。"""
    import subprocess
    import sys

    print("========== 环境自检(2×T4) ==========")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print('torch', torch.__version__, 'gpus', torch.cuda.device_count())",
        ],
        check=False,
    )

    print("\n========== torchrun --nproc_per_node=2 verify_lora_tp.py ==========")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29555",
            REMOTE_TP_SCRIPT,
        ],
        check=False,
    )
    print(f"\n[exit] verify_lora_tp.py returncode={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


@app.local_entrypoint()
def main() -> None:
    run_verify.remote()


@app.local_entrypoint()
def tp() -> None:
    run_verify_tp.remote()
