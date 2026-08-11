"""在容器里对两个 Relax PR worktree 跑 ruff（本地 pip 连不上源）。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_lint_relax_prs.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
PR1_DIR = HERE.parent / "_relax_pr1"
PR2_DIR = HERE.parent / "_relax_pr2"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

TARGETS = [
    "relax/utils/megatron_peft_utils.py",
    "tests/utils/test_megatron_peft_utils.py",
]

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .pip_install("ruff")
    .add_local_dir(PR1_DIR.as_posix(), "/root/pr1", copy=True)
    .add_local_dir(PR2_DIR.as_posix(), "/root/pr2", copy=True)
)

app = modal.App("relax-pr-lint")


@app.function(image=image, timeout=15 * 60)
def lint() -> int:
    import subprocess
    import sys

    failed = []
    for name, workdir in (("PR1", "/root/pr1"), ("PR2", "/root/pr2")):
        for tool_args in (["format", "--check"], ["check"]):
            tag = f"{name} ruff {' '.join(tool_args)}"
            print(f"\n===== {tag} =====", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "ruff", *tool_args, *TARGETS],
                cwd=workdir,
                check=False,
            )
            print(f"[{tag}] return code = {proc.returncode}", flush=True)
            if proc.returncode != 0:
                failed.append(tag)

    print("\n===== 汇总 =====", flush=True)
    if failed:
        for tag in failed:
            print(f"  [BAD] {tag}", flush=True)
        return 1
    print("  [OK] 两个 worktree 的格式与 lint 都干净", flush=True)
    return 0


@app.local_entrypoint()
def main() -> None:
    rc = lint.remote()
    if rc != 0:
        raise SystemExit(f"ruff 未通过（return={rc}）")
