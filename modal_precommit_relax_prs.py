"""在容器里对两个 Relax PR worktree 跑仓库自带的 pre-commit。

CI 的 Pre-commit Checks 在 #262 上挂了，日志要登录才能下，所以本地复现。
worktree 拷进容器后 .git 是个失效的指针文件，先 git init 再跑。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_precommit_relax_prs.py
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

CHANGED = [
    "relax/utils/megatron_peft_utils.py",
    "tests/utils/test_megatron_peft_utils.py",
]

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .apt_install("git")
    .pip_install("pre-commit")
    .add_local_dir(PR1_DIR.as_posix(), "/root/pr1", copy=True)
    .add_local_dir(PR2_DIR.as_posix(), "/root/pr2", copy=True)
)

app = modal.App("relax-pr-precommit")


@app.function(image=image, timeout=30 * 60)
def run_precommit() -> int:
    import shutil
    import subprocess
    import sys

    def sh(args, cwd, check=True):
        return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)

    results = {}
    for name, workdir in (("PR1", "/root/pr1"), ("PR2", "/root/pr2")):
        print(f"\n########## {name} ##########", flush=True)

        # worktree 的 .git 是指针文件，容器里失效；重建一个干净仓库
        dotgit = os.path.join(workdir, ".git")
        if os.path.isfile(dotgit):
            os.remove(dotgit)
        elif os.path.isdir(dotgit):
            shutil.rmtree(dotgit)
        sh(["git", "init", "-q"], workdir)
        sh(["git", "config", "user.email", "ci@example.com"], workdir)
        sh(["git", "config", "user.name", "ci"], workdir)
        sh(["git", "add", "-A"], workdir)
        sh(["git", "commit", "-q", "-m", "base"], workdir)

        proc = subprocess.run(
            [sys.executable, "-m", "pre_commit", "run", "--files", *CHANGED],
            cwd=workdir,
            check=False,
        )
        print(f"[{name}] pre-commit return code = {proc.returncode}", flush=True)
        results[name] = proc.returncode

        if proc.returncode != 0:
            diff = sh(["git", "diff"], workdir, check=False).stdout
            print(f"\n----- {name} 被 hook 改动的内容 -----\n{diff}", flush=True)

    print("\n===== 汇总 =====", flush=True)
    for name, code in results.items():
        print(f"  {name}: return={code}", flush=True)
    return 0 if all(c == 0 for c in results.values()) else 1


@app.local_entrypoint()
def main() -> None:
    rc = run_precommit.remote()
    print(f"\n整体 return={rc}", flush=True)
