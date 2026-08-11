"""在 v2 镜像里跑两个 LoRA 单测。

- test_should_apply_lora_gate.py    通用 gate 行为，将来随 PR 提给 sgl-project
- test_qwen3_omni_lora_pattern.py   Omni 的 _lora_pattern，属于我们的 delta

sglang 源码从 fork 的 lora-omni-v2 分支 clone（带 gate 修复），
两个测试文件用本地的，方便改完直接验，不用先推一次。

用法：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_run_lora_tests.py
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

SGLANG_REPO = "https://github.com/SakaiXue6666/sglang.git"
SGLANG_REF = os.environ.get("SGLANG_V2_REF", "lora-omni-v2")
SGLANG_SRC = "/root/SglangSrc"

GATE_TEST = "test/registered/unit/lora/test_should_apply_lora_gate.py"
OMNI_TEST = "test/registered/unit/models/test_qwen3_omni_lora_pattern.py"

test_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {SGLANG_REF} {SGLANG_REPO} {SGLANG_SRC}",
    )
    .pip_install("pytest")
    .add_local_file(
        (HERE / "sglang" / GATE_TEST.replace("/", os.sep)).as_posix(),
        f"{SGLANG_SRC}/{GATE_TEST}",
        copy=True,
    )
    .add_local_file(
        (HERE / "sglang" / OMNI_TEST.replace("/", os.sep)).as_posix(),
        f"{SGLANG_SRC}/{OMNI_TEST}",
        copy=True,
    )
    .env({"PYTHONPATH": f"{SGLANG_SRC}/python:/root/Megatron-LM/:/pkg/:/root/"})
)

app = modal.App("relax-omni-lora-unit-tests")


@app.function(image=test_image, gpu="T4", timeout=30 * 60)
def run_tests() -> int:
    import subprocess
    import sys

    subprocess.run(
        ["git", "log", "-1", "--format=sglang %H %s"], cwd=SGLANG_SRC, check=False
    )
    subprocess.run(
        [sys.executable, "-c", "import sglang; print('sglang 加载自:', sglang.__file__)"],
        check=False,
    )

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", GATE_TEST, OMNI_TEST, "-v", "--no-header"],
        cwd=SGLANG_SRC,
        check=False,
    )
    print(f"\n[exit] pytest returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = run_tests.remote()
    if rc != 0:
        raise SystemExit(f"单测未通过 (returncode={rc})")
    print("\n[TESTS OK] gate 与 Omni pattern 单测全部通过。", flush=True)
