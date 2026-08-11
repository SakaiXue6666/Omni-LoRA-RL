"""在 v2 镜像里验证两个 Relax PR 分支。

每个分支跑两遍：打了补丁的版本要全过；把 relax/utils/megatron_peft_utils.py
换回上游 main 的原版后，新加的用例要挂 —— 这样才能证明测试确实咬住了改动。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_verify_relax_prs.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
PR1_DIR = HERE.parent / "_relax_pr1"
PR2_DIR = HERE.parent / "_relax_pr2"
PRISTINE = HERE / "Relax" / "relax" / "utils" / "megatron_peft_utils.py"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

TEST_FILES = [
    "tests/utils/test_megatron_peft_utils.py",
    "tests/backends/megatron/weight_update/test_lora_weight_sync.py",
]

# 只属于本 PR 的新用例，用来做「换回原版就挂」的负向对照。
PR1_NEW = "pattern or exact_paths or trailing_wildcard"
PR2_NEW = "peft_prefix or TestToPeftStateDict"

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .pip_install("pytest", "pytest-asyncio")
    .add_local_dir(PR1_DIR.as_posix(), "/root/pr1", copy=True)
    .add_local_dir(PR2_DIR.as_posix(), "/root/pr2", copy=True)
    .add_local_file(PRISTINE.as_posix(), "/root/pristine_megatron_peft_utils.py", copy=True)
)

app = modal.App("relax-pr-verify")


@app.function(image=image, timeout=30 * 60)
def verify() -> int:
    import shutil
    import subprocess
    import sys

    module_rel = "relax/utils/megatron_peft_utils.py"

    def run(tag: str, workdir: str, args: list[str]) -> int:
        print(f"\n===== {tag} =====", flush=True)
        env = dict(os.environ, PYTHONPATH=workdir)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *args, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=workdir,
            env=env,
            check=False,
        )
        print(f"[{tag}] return code = {proc.returncode}", flush=True)
        return proc.returncode

    results: dict[str, int] = {}

    for name, workdir, selector in (("PR1", "/root/pr1", PR1_NEW), ("PR2", "/root/pr2", PR2_NEW)):
        module_path = os.path.join(workdir, module_rel)
        backup = module_path + ".patched"

        results[f"{name} 打补丁"] = run(f"{name} 打补丁", workdir, TEST_FILES)

        shutil.copyfile(module_path, backup)
        shutil.copyfile("/root/pristine_megatron_peft_utils.py", module_path)
        results[f"{name} 换回原版"] = run(f"{name} 换回原版", workdir, [TEST_FILES[0], "-k", selector])
        shutil.copyfile(backup, module_path)

    print("\n===== 汇总 =====", flush=True)
    ok = True
    for tag, code in results.items():
        want_pass = tag.endswith("打补丁")
        passed = code == 0
        verdict = "OK" if passed == want_pass else "BAD"
        if verdict == "BAD":
            ok = False
        expect = "应当全过" if want_pass else "应当失败"
        print(f"  [{verdict}] {tag}: return={code}（{expect}）", flush=True)
    return 0 if ok else 1


@app.local_entrypoint()
def main() -> None:
    rc = verify.remote()
    if rc != 0:
        raise SystemExit(f"验证未通过（return={rc}）")
    print("\n两个 PR 分支都验证通过。", flush=True)
