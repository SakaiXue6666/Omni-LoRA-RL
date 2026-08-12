"""验证 Relax 第三个 PR：pytest + 仓库自带 pre-commit。

判据不是"跑绿了"，而是可证伪：把 serialize_adapter_tensors 换回上游那种共享内存写法，
新单测必须挂；换回来必须过。所以这里跑两遍。

pre-commit 那段是 #262 的教训——ruff 全过、CI 仍挂在 docformatter 的 79 列重排上，
所以推之前先在容器里复现一次。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_verify_pr3.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
PR3_DIR = HERE.parent / "_relax_pr3"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

CHANGED = [
    "relax/utils/megatron_peft_utils.py",
    "relax/backends/megatron/weight_update/update_weight_from_tensor.py",
    "tests/utils/test_megatron_peft_utils.py",
]

TEST_TARGET = "tests/utils/test_megatron_peft_utils.py::TestSerializeAdapterTensors"

# 上游的写法：payload 里放的是 /dev/shm 引用而不是字节。用它替换掉函数体，
# 新单测里"payload 必须扛得住生产者撒手""长度必须够装下张量"两条应当失败。
UPSTREAM_BODY = '''
def serialize_adapter_tensors(tensors):
    """上游写法：共享内存引用（仅用于 A/B，故意不内联字节）。"""
    from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy

    from sglang.srt.utils import MultiprocessingSerializer

    prev = get_sharing_strategy()
    set_sharing_strategy("file_system")
    try:
        return MultiprocessingSerializer.serialize(dict(tensors), output_str=True)
    finally:
        set_sharing_strategy(prev)
'''

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .apt_install("git")
    .pip_install("pre-commit")
    .add_local_dir(PR3_DIR.as_posix(), "/root/pr3", copy=True)
)

app = modal.App("relax-pr3-verify")

WORK = "/root/pr3"
UTILS = f"{WORK}/relax/utils/megatron_peft_utils.py"


@app.function(image=image, timeout=30 * 60)
def verify() -> int:
    import re
    import shutil
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=WORK)

    def pytest_run(label: str) -> int:
        print(f"\n########## pytest（{label}） ##########", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", TEST_TARGET, "-q", "--no-header"],
            cwd=WORK,
            env=env,
            check=False,
        )
        print(f"[{label}] return={proc.returncode}", flush=True)
        return proc.returncode

    with open(UTILS, encoding="utf-8") as f:
        patched_source = f.read()

    rc_patched = pytest_run("本 PR 的实现")

    # A/B：换成上游写法，新单测必须挂
    upstream_source = re.sub(
        r"\ndef serialize_adapter_tensors.*?\n\n\n__all__",
        UPSTREAM_BODY + "\n\n__all__",
        patched_source,
        flags=re.DOTALL,
    )
    if upstream_source == patched_source:
        print("!! 没能替换掉函数体，A/B 作废", flush=True)
        return 1
    with open(UTILS, "w", encoding="utf-8") as f:
        f.write(upstream_source)
    rc_upstream = pytest_run("换回上游的共享内存写法")

    with open(UTILS, "w", encoding="utf-8") as f:
        f.write(patched_source)

    print("\n########## pre-commit ##########", flush=True)
    dotgit = os.path.join(WORK, ".git")
    if os.path.isfile(dotgit):
        os.remove(dotgit)
    elif os.path.isdir(dotgit):
        shutil.rmtree(dotgit)
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "ci@example.com"],
        ["git", "config", "user.name", "ci"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "base"],
    ):
        subprocess.run(args, cwd=WORK, check=True, capture_output=True, text=True)
    hooks = subprocess.run(
        [sys.executable, "-m", "pre_commit", "run", "--files", *CHANGED],
        cwd=WORK,
        check=False,
    )
    print(f"[pre-commit] return={hooks.returncode}", flush=True)
    if hooks.returncode != 0:
        diff = subprocess.run(["git", "diff"], cwd=WORK, check=False, capture_output=True, text=True)
        print(f"\n----- 被 hook 改动的内容 -----\n{diff.stdout}", flush=True)

    print("\n===== 汇总 =====", flush=True)
    print(f"  本 PR 的实现:        {'过' if rc_patched == 0 else '挂'}（应当过）", flush=True)
    print(f"  上游共享内存写法:    {'过' if rc_upstream == 0 else '挂'}（应当挂）", flush=True)
    print(f"  pre-commit:          {'过' if hooks.returncode == 0 else '挂'}（应当过）", flush=True)

    ok = rc_patched == 0 and rc_upstream != 0 and hooks.returncode == 0
    print(f"  判定：{'通过' if ok else '不通过'}", flush=True)
    return 0 if ok else 1


@app.local_entrypoint()
def main() -> None:
    print(f"\n[verify] 退出码 = {verify.remote()}")
