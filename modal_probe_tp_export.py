"""探针八 —— 在 2 张卡上验 TP=2 的 adapter 导出 parity。

用 torchrun 起 2 个进程，其余（镜像、Relax 源码指针、模型卷）与探针七一致。

跑法（PowerShell）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_tp_export.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_09_tp_export.py"
REMOTE_SCRIPT = "/root/mig_09_tp_export.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

RELAX_REPO = "https://github.com/redai-infra/Relax.git"
RELAX_REF = os.environ.get("RELAX_V2_REF", "9a5674afde12f608698ab4f60cdb9849a0eb6cb3")
RELAX_SRC = "/root/RelaxSrc"

GPU = os.environ.get("PROBE_GPU", "A10G:2")
TP_SIZE = os.environ.get("PROBE_TP_SIZE", "2")

probe_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none {RELAX_REPO} {RELAX_SRC}",
        f"cd {RELAX_SRC} && git checkout {RELAX_REF} && git log -1 --format='RELAX_HEAD %H %cI %s'",
    )
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .env(
        {
            "OMNI_CKPT": OMNI_CKPT,
            "PROBE_TP_SIZE": TP_SIZE,
            "PYTHONPATH": f"{RELAX_SRC}:/root/Megatron-LM/:/pkg/:/root/",
        }
    )
)

app = modal.App("relax-omni-probe-tp-export")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=probe_image, gpu=GPU, timeout=40 * 60, volumes={MODEL_MOUNT: model_volume})
def probe() -> int:
    import subprocess
    import sys

    import torch

    print(f"========== 可见 GPU: {torch.cuda.device_count()} ==========", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}", flush=True)

    print("\n========== torchrun --nproc_per_node=2 ==========", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={TP_SIZE}",
        "--master_port=29597",
        REMOTE_SCRIPT,
    ]
    proc = subprocess.run(cmd, check=False)
    print(f"\n[exit] mig_09_tp_export.py return code = {proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针八失败（return code={rc}）")
    print("\n[PROBE DONE] TP=2 导出 parity 通过。", flush=True)
