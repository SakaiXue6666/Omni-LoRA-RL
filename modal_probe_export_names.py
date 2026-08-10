"""探针二 —— 在 v2 镜像上验证 adapter 导出命名与 SGLang 的 parity。

为什么便宜：
    全程不建 Megatron 模型、不加载权重。命名由真实的 mapping_registry 推导，
    SGLang 侧喂的是按真实 config 算好形状的零张量。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_export_names.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_03_export_names.py"
REMOTE_SCRIPT = "/root/mig_03_export_names.py"

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

app = modal.App("relax-omni-probe-export-names")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=probe_image, gpu="T4", timeout=30 * 60, volumes={MODEL_MOUNT: model_volume})
def probe() -> int:
    import subprocess
    import sys

    print("========== Relax 源码版本 ==========", flush=True)
    subprocess.run(
        ["git", "log", "-1", "--format=commit %H%ndate   %cI%nsubject %s"],
        cwd=RELAX_SRC,
        check=False,
    )

    print("\n========== 运行 mig_03_export_names.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_03_export_names.py returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针二未通过 (returncode={rc})")
    print("\n[PROBE DONE] 命名 parity 已验证。", flush=True)
