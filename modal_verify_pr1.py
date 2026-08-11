"""验证要提给 sgl-project 的第一个 PR 分支。

两件事：
    1. 在上游 main + 这个补丁上，新加的单测能过
    2. 把补丁那几行去掉后，单测必须挂 —— 否则这个测试没有意义

用法：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_verify_pr1.py
"""

from __future__ import annotations

import os

import modal

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

FORK = "https://github.com/SakaiXue6666/sglang.git"
BRANCH = os.environ.get("PR1_BRANCH", "fix/lora-honor-should-apply-lora")
SRC = "/root/SglangPR"

TEST = "test/registered/unit/lora/test_should_apply_lora_gate.py"
GATE_ANCHOR = "should_apply_lora = getattr(self.base_model, \"should_apply_lora\", None)"

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {BRANCH} {FORK} {SRC}",
        f"cd {SRC} && git log -1 --format='PR_HEAD %H %s'",
    )
    .pip_install("pytest")
    .env({"PYTHONPATH": f"{SRC}/python:/pkg/:/root/"})
)

app = modal.App("sglang-pr1-verify")


@app.function(image=image, gpu="T4", timeout=40 * 60)
def verify() -> int:
    import pathlib
    import subprocess
    import sys

    def run_test(tag: str) -> int:
        print(f"\n========== pytest（{tag}）==========", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TEST, "-v", "--no-header", "-x"],
            cwd=SRC,
            check=False,
        )
        print(f"[{tag}] returncode={proc.returncode}", flush=True)
        return proc.returncode

    subprocess.run(["git", "log", "-1", "--format=%H %s"], cwd=SRC, check=False)

    rc_with_gate = run_test("带补丁")

    # 把门的两行注释掉，模拟上游现状
    manager = pathlib.Path(SRC) / "python/sglang/srt/lora/lora_manager.py"
    text = manager.read_text(encoding="utf-8")
    assert GATE_ANCHOR in text, "没找到补丁代码，分支不对？"
    lines = text.split("\n")
    out = []
    skip = 0
    for line in lines:
        if GATE_ANCHOR in line:
            skip = 3  # 赋值 + if + continue
        if skip > 0:
            skip -= 1
            out.append("            # removed for the negative check")
            continue
        out.append(line)
    manager.write_text("\n".join(out), encoding="utf-8")
    print("\n已移除门的调用点，重新跑一遍", flush=True)

    rc_without_gate = run_test("去掉补丁")

    print("\n========== 结论 ==========", flush=True)
    ok = True
    if rc_with_gate != 0:
        print("  [FAIL] 带补丁时单测没过", flush=True)
        ok = False
    else:
        print("  [PASS] 带补丁时单测通过", flush=True)
    if rc_without_gate == 0:
        print("  [FAIL] 去掉补丁后单测仍然通过 —— 这个测试测不到东西", flush=True)
        ok = False
    else:
        print("  [PASS] 去掉补丁后单测失败 —— 测试确实盯住了这个行为", flush=True)
    return 0 if ok else 1


@app.local_entrypoint()
def main() -> None:
    rc = verify.remote()
    if rc != 0:
        raise SystemExit(f"PR 分支验证未通过 (returncode={rc})")
    print("\n[PR1 OK] 分支可以提交。", flush=True)
