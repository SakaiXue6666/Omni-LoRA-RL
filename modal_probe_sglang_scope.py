"""探针四 —— 在 v2 镜像上验证 sglang 侧的 LoRA 作用范围。

关键点：跑的是我们 fork 里改过的 sglang 源码（lora-omni-v2 分支），
不是镜像里预装的那份，否则验不到新加的 should_apply_lora 门。
源码目录挂到 PYTHONPATH 最前面，编译好的 sgl_kernel 仍用镜像里的。

用法：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_sglang_scope.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_05_sglang_lora_scope.py"
REMOTE_SCRIPT = "/root/mig_05_sglang_lora_scope.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

SGLANG_REPO = "https://github.com/SakaiXue6666/sglang.git"
SGLANG_REF = os.environ.get("SGLANG_V2_REF", "lora-omni-v2")
SGLANG_SRC = "/root/SglangSrc"

probe_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {SGLANG_REF} {SGLANG_REPO} {SGLANG_SRC}",
        f"cd {SGLANG_SRC} && git log -1 --format='SGLANG_HEAD %H %cI %s'",
    )
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .env(
        {
            "OMNI_CKPT": OMNI_CKPT,
            # fork 的源码必须排在预装 sglang 之前
            "PYTHONPATH": f"{SGLANG_SRC}/python:/root/Megatron-LM/:/pkg/:/root/",
        }
    )
)

app = modal.App("relax-omni-probe-sglang-scope")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=probe_image, gpu="T4", timeout=40 * 60, volumes={MODEL_MOUNT: model_volume})
def probe() -> int:
    import subprocess
    import sys

    print("========== 确认用的是 fork 的源码 ==========", flush=True)
    subprocess.run(
        ["git", "log", "-1", "--format=commit %H%ndate   %cI%nsubject %s"],
        cwd=SGLANG_SRC,
        check=False,
    )
    subprocess.run(
        [sys.executable, "-c", "import sglang; print('sglang 实际加载自:', sglang.__file__)"],
        check=False,
    )

    print("\n========== 运行 mig_05_sglang_lora_scope.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针四未通过 (returncode={rc})")
    print("\n[PROBE DONE] sglang 侧 LoRA 作用范围已验证。", flush=True)
