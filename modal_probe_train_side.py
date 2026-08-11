"""探针七 —— 在 v2 镜像上跑训练侧冒烟（Bridge 建 Omni + LoRA 注入 + adapter 导出）。

镜像与探针一同一个 digest，Relax 源码钉到 v2 submodule 指针。缩层之后一张 A10G
就够，不必上 A100。

跑法（PowerShell）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_train_side.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_08_train_side.py"
REMOTE_SCRIPT = "/root/mig_08_train_side.py"

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

GPU = os.environ.get("PROBE_GPU", "A10G")

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
            "PYTHONPATH": f"{RELAX_SRC}:/root/Megatron-LM/:/pkg/:/root/",
        }
    )
)

app = modal.App("relax-omni-probe-train-side")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=probe_image, gpu=GPU, timeout=40 * 60, volumes={MODEL_MOUNT: model_volume})
def probe() -> int:
    import subprocess
    import sys

    print("========== Relax 源码指针 ==========", flush=True)
    subprocess.run(
        ["git", "log", "-1", "--format=commit %H%nauthor %ci%nsubject %s"],
        cwd=RELAX_SRC,
        check=False,
    )

    print("\n========== 跑 mig_08_train_side.py ==========", flush=True)
    proc = subprocess.run([sys.executable, "-u", REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_08_train_side.py return code = {proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针七失败（return code={rc}）")
    print("\n[PROBE DONE] 训练侧冒烟通过。", flush=True)
