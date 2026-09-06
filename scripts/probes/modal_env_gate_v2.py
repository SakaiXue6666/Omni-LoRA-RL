"""v2 迁移的两个前置探测：新版 Relax 镜像的环境闸门 + v1 镜像定位。

对应 MIGRATION_V2 计划的第 0 节与 v1 冻结清单。

一、gate_relax —— 环境闸门（决定迁移成不成立）
    2026-07 的迁移之所以暂停，是因为往 slimerl/slime 镜像上 pip 嫁接 megatron-bridge 0.5.0
    两轮都失败（第一轮 megatron.core 太旧缺 safe_get_world_size；第二轮 flashinfer 版本冲突 +
    transformers/peft 半装）。当时的结论是必须换一个自洽的基础镜像。

    新版 Relax 官方镜像正是那个自洽镜像：它的 Dockerfile 从 lmsysorg/sglang 起，clone
    NVIDIA-NeMo/Megatron-Bridge 钉死 commit 并用 3rdparty/Megatron-LM submodule 提供
    megatron.core（不是 pip 嫁接）。本函数就是把当年那 4 个判据在这个镜像上重跑一遍。

    与旧 modal_migrate.py 的区别：不再有 BRIDGE_INSTALL 那一步，bridge 由镜像自带。

二、v1_versions_* —— 定位 v1 到底用的是哪个 slime 镜像
    v1 的 BASE_IMAGE 写的是 slimerl/slime:latest，而该 tag 会移动：实测 2026-08-10 查到的
    latest 构建于当天 19:31，而 v1 实验窗口是 7/8-7/24，环境已漂移，v1 当前不可复现。
    slime 有日期 nightly tag，窗口两侧各有一个候选，用版本表和 MIGRATION_PLAN.md 里记录的
    flashinfer 0.6.3 / transformers 4.57.1 比对即可判定。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_env_gate_v2.py                 # 全部跑
    modal run modal_env_gate_v2.py --mode gate     # 只跑环境闸门
    modal run modal_env_gate_v2.py --mode v1       # 只跑 v1 镜像定位
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_00_env.py"
REMOTE_SCRIPT = "/root/mig_00_env.py"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"
OMNI_CKPT = "/models/qwen3-omni"

# 用 digest 而非可变 tag 锁镜像（MIGRATION_PLAN.md 的既定纪律）。
# ghcr.io/redai-infra/relaxrl:latest == dev-20260723-8cc1e8fd，解析于 2026-08-10。
RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

# v1 镜像候选，digest 解析于 2026-08-10。
V1_IMAGE_0707 = "slimerl/slime@sha256:a7317182c71d35712ee4edc86a5d1c313dc969efdf0026d339673299c186ea75"
V1_IMAGE_0722 = "slimerl/slime@sha256:a97ec147e37bef050337a9b229036eda00b4aa9c4d02b31a0109dc850f8ca342"

# 需要报版本的包。flashinfer 三件套必须同版本，是 v1 第二轮失败的直接原因，
# 也是区分两个候选镜像的主要判据。
PROBE_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "megatron-core",
    "megatron-bridge",
    "transformer-engine",
    "transformer-engine-torch",
    "flashinfer",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "sglang",
    "sgl-kernel",
    "ray",
    "numpy",
    "accelerate",
    "safetensors",
)

# 在容器里跑的版本 dump。用 importlib.metadata 读元数据而不 import 模块本体，
# 避免 transformer_engine 这类包在无 GPU 环境下 import 失败导致整个探测中断。
_VERSION_DUMP = r"""
import sys
from importlib.metadata import distributions, version, PackageNotFoundError

WANTED = {wanted!r}

print("PYTHON:", sys.version.replace("\n", " "), flush=True)
print("========== 关键包版本 ==========", flush=True)
for name in WANTED:
    try:
        print(f"  {{name}}: {{version(name)}}", flush=True)
    except PackageNotFoundError:
        print(f"  {{name}}: <未安装>", flush=True)

print("========== 全部含 megatron/flash/sglang/transformer 的发行包 ==========", flush=True)
for dist in sorted(distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
    name = (dist.metadata["Name"] or "").lower()
    if any(k in name for k in ("megatron", "flash", "sglang", "sgl-", "transformer", "torch")):
        print(f"  {{dist.metadata['Name']}}: {{dist.version}}", flush=True)
""".format(wanted=list(PROBE_PACKAGES))


def _dump_versions(label: str) -> None:
    import subprocess
    import sys

    print(f"\n########## {label} ##########", flush=True)
    subprocess.run([sys.executable, "-c", _VERSION_DUMP], check=False)


gate_image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .env({"OMNI_CKPT": OMNI_CKPT})
)

v1_image_0707 = modal.Image.from_registry(V1_IMAGE_0707, add_python=None)
v1_image_0722 = modal.Image.from_registry(V1_IMAGE_0722, add_python=None)

app = modal.App("relax-omni-env-gate-v2")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=gate_image, gpu="T4", timeout=30 * 60, volumes={MODEL_MOUNT: model_volume})
def gate_relax() -> int:
    """环境闸门：在新版 Relax 官方镜像上重跑 2026-07 卡住的 4 个判据。"""
    import subprocess
    import sys

    _dump_versions("Relax 官方镜像版本表")

    print("\n========== Megatron-LM 来源确认 ==========", flush=True)
    # Relax 的 Dockerfile 把 bridge 的 3rdparty/Megatron-LM rsync 到 /root/Megatron-LM，
    # 并通过 PYTHONPATH 生效。确认 megatron.core 确实来自这里而非 pip 嫁接的版本。
    subprocess.run(["ls", "-la", "/root/Megatron-LM"], check=False)
    print("PYTHONPATH =", os.environ.get("PYTHONPATH"), flush=True)

    print("\n========== 运行 mig_00_env.py ==========", flush=True)
    proc = subprocess.run([sys.executable, REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_00_env.py returncode={proc.returncode}", flush=True)
    return proc.returncode


@app.function(image=v1_image_0707, timeout=15 * 60)
def v1_versions_0707() -> None:
    """v1 候选镜像 A：nightly-dev-20260707a（对应 7/7-7/8 的 Phase 0.1 探测）。"""
    _dump_versions("slime nightly-dev-20260707a")


@app.function(image=v1_image_0722, timeout=15 * 60)
def v1_versions_0722() -> None:
    """v1 候选镜像 B：nightly-dev-20260722a（对应 7/23-7/24 的 talker 实验）。"""
    _dump_versions("slime nightly-dev-20260722a")


@app.local_entrypoint()
def main(mode: str = "all") -> None:
    if mode in ("all", "v1"):
        # 两个候选并行跑，省时间。
        handles = [v1_versions_0707.spawn(), v1_versions_0722.spawn()]
        for h in handles:
            h.get()

    if mode in ("all", "gate"):
        rc = gate_relax.remote()
        if rc != 0:
            raise SystemExit(f"环境闸门未通过 (mig_00_env.py returncode={rc})")
        print("\n[GATE OK] 新版 Relax 镜像通过 Phase 0.1 判据，迁移成立。", flush=True)
