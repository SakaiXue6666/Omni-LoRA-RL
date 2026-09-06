"""探针三 —— PEFT 前缀契约。纯 CPU，复用已缓存的 v2 镜像，只多装一个 peft。

用法：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_peft_prefix.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_04_peft_prefix.py"
REMOTE_SCRIPT = "/root/mig_04_peft_prefix.py"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

probe_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .pip_install("peft")
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
)

app = modal.App("relax-omni-probe-peft-prefix")


@app.function(image=probe_image, timeout=20 * 60)
def probe() -> int:
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_04_peft_prefix.py returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = probe.remote()
    if rc != 0:
        raise SystemExit(f"探针三未通过 (returncode={rc})")
