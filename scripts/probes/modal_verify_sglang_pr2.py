"""验证要提给 sgl-project 的第二个 PR：patch_torch 的 CPU 张量守卫。

三件事，缺一不可：
    1. 带补丁时新单测过
    2. 把守卫去掉（还原成上游现状）后必须挂 —— 否则测试测不到东西
    3. 上游的 check_registered_tests.py 与 isort/black 干净

纯 CPU 就够：这个 bug 本身就是 CPU 张量触发的。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_verify_sglang_pr2.py
"""

from __future__ import annotations

import os

import modal


RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

FORK = "https://github.com/SakaiXue6666/sglang.git"
BRANCH = os.environ.get("PR2_BRANCH", "fix/reduce-tensor-cpu-guard")
SRC = "/root/SglangPR2"

TEST = "test/registered/unit/utils/test_patch_torch_cpu_tensor.py"
PATCHED = "python/sglang/srt/utils/patch_torch.py"

# 上游原样：无条件改写第 6 位
UPSTREAM = """def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = reductions._reduce_tensor_original(*args, **kwargs)
    output_args = _modify_tuple(
        output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
    )
    return output_fn, output_args"""

def _head_sha() -> str:
    """把 worktree 的 HEAD 写进构建命令，否则 Modal 会拿缓存层里的旧克隆来验。"""
    import pathlib
    import subprocess

    wt = pathlib.Path(__file__).resolve().parent.parent / "_sglang_pr2"
    out = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# 容器里也会 import 这个文件，那边没有 worktree，所以只在本地算。
SHA = _head_sha() if modal.is_local() else os.environ.get("PR_SHA", "")

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {BRANCH} {FORK} {SRC}",
        f"cd {SRC} && git checkout {SHA} && git log -1 --format='PR_HEAD %H %s'",
    )
    .pip_install("pytest", "isort", "black")
    .env({"PYTHONPATH": f"{SRC}/python:/pkg/:/root/", "PR_SHA": SHA})
)

app = modal.App("sglang-pr2-verify")


@app.function(image=image, cpu=2.0, timeout=30 * 60)
def verify() -> int:
    import pathlib
    import re
    import subprocess
    import sys

    def run_test(tag: str) -> int:
        print(f"\n========== pytest（{tag}）==========", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TEST, "-v", "--no-header"],
            cwd=SRC,
            check=False,
        )
        print(f"[{tag}] returncode={proc.returncode}", flush=True)
        return proc.returncode

    rc_patched = run_test("带守卫")

    target = pathlib.Path(SRC) / PATCHED
    patched_text = target.read_text(encoding="utf-8")
    upstream_text = re.sub(
        r"def _reduce_tensor_modified.*?return output_fn, output_args",
        UPSTREAM,
        patched_text,
        flags=re.DOTALL,
        count=1,
    )
    if upstream_text == patched_text:
        print("!! 没能还原成上游写法，A/B 作废", flush=True)
        return 1
    target.write_text(upstream_text, encoding="utf-8")
    rc_upstream = run_test("还原成上游")
    target.write_text(patched_text, encoding="utf-8")

    print("\n========== 上游的测试注册检查 ==========", flush=True)
    reg = subprocess.run(
        [sys.executable, "scripts/ci/check_registered_tests.py"], cwd=SRC, check=False
    )
    print(f"[check_registered_tests] returncode={reg.returncode}", flush=True)

    print("\n========== isort / black ==========", flush=True)
    fmt = 0
    for tool, args in (("isort", ["--check-only", "--diff"]), ("black", ["--check", "--diff"])):
        proc = subprocess.run(
            [sys.executable, "-m", tool, *args, PATCHED, TEST], cwd=SRC, check=False
        )
        print(f"[{tool}] returncode={proc.returncode}", flush=True)
        fmt |= proc.returncode

    print("\n========== 结论 ==========", flush=True)
    checks = {
        "带守卫时单测通过": rc_patched == 0,
        "还原上游后单测失败": rc_upstream != 0,
        "测试注册检查通过": reg.returncode == 0,
        "isort/black 干净": fmt == 0,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    return 0 if all(checks.values()) else 1


@app.local_entrypoint()
def main() -> None:
    rc = verify.remote()
    if rc != 0:
        raise SystemExit(f"分支验证未通过 (returncode={rc})")
    print("\n[PR2 OK] 分支可以提交。", flush=True)
