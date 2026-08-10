"""探针一 —— 在 v2 镜像上确认 LoRA 的作用范围。

为什么便宜：
    塔用 meta device 构建，不加载权重、层数压到 1，不建 Megatron 模型、
    不初始化分布式。GPU 只是为了让 megatron / TE 的 import 走通常路径，
    实际算力几乎不用，T4 跑几分钟即可。

Relax 源码钉在 v2 submodule 的同一个 commit 上，保证探针结论对应 v2 的代码。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_lora_scope.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_02_lora_scope.py"
REMOTE_SCRIPT = "/root/mig_02_lora_scope.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

# 与 Phase 0.1 / 0.1b 同一个 digest。
RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

RELAX_REPO = "https://github.com/redai-infra/Relax.git"
# 钉死到 v2 的 Relax submodule 指针，而不是 main —— main 会漂。
RELAX_REF = os.environ.get("RELAX_V2_REF", "9a5674afde12f608698ab4f60cdb9849a0eb6cb3")
RELAX_SRC = "/root/RelaxSrc"

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
            # 镜像原有 PYTHONPATH 必须保留，megatron.core 来自 /root/Megatron-LM。
            "PYTHONPATH": f"{RELAX_SRC}:/root/Megatron-LM/:/pkg/:/root/",
        }
    )
)

app = modal.App("relax-omni-probe-lora-scope")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=probe_image, gpu="T4", timeout=20 * 60, volumes={MODEL_MOUNT: model_volume})
def probe() -> int:
    import subprocess
    import sys

    print("========== Relax 源码版本 ==========", flush=True)
    subprocess.run(
        ["git", "log", "-1", "--format=commit %H%ndate   %cI%nsubject %s"],
        cwd=RELAX_SRC,
        check=False,
    )

    print("\n========== 运行 mig_02_lora_scope.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_02_lora_scope.py returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针一未跑完 (returncode={rc})")
    print("\n[PROBE DONE] 见上方 [5] 的命中情况。", flush=True)
