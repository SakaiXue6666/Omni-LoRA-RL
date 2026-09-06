"""探针五 —— 在 1 张 A100 上把 v2 的 sglang 带 LoRA 跑起来。

复用 v1 就有的 qwen3-omni-weights 卷（/models/qwen3-omni），不重新下权重。
sglang 用 fork 的 lora-omni-v2 分支源码顶到 PYTHONPATH 最前面，确保验的是
我们的改动而不是镜像里预装的那份。

    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_probe_serve_lora.py
"""

from __future__ import annotations

import os
import pathlib

import modal


HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SCRIPT = HERE / "mig_06_serve_lora.py"
REMOTE_SCRIPT = "/root/mig_06_serve_lora.py"
LOCAL_REDUCE = HERE / "mig_07_cpu_reduce.py"
REMOTE_REDUCE = "/root/mig_07_cpu_reduce.py"

RELAX_IMAGE = os.environ.get(
    "RELAX_V2_IMAGE",
    "ghcr.io/redai-infra/relaxrl@sha256:8dc39af377a570e6cd7ec88c8b7fcd44c1eb820111e9d2069f1c7c3024b2ea23",
)

SGLANG_FORK = "https://github.com/SakaiXue6666/sglang.git"
SGLANG_REF = os.environ.get("SGLANG_V2_REF", "lora-omni-v2")
SGLANG_SRC = "/root/sglangSrc"

MODEL_VOLUME = "qwen3-omni-weights"
MODEL_MOUNT = "/models"

image = (
    modal.Image.from_registry(RELAX_IMAGE, add_python=None)
    .run_commands(
        f"git clone --filter=blob:none --branch {SGLANG_REF} {SGLANG_FORK} {SGLANG_SRC}",
        f"cd {SGLANG_SRC} && git log -1 --format='%H %s'",
    )
    .env({"PYTHONPATH": f"{SGLANG_SRC}/python:/root:/sgl-workspace/sglang/python"})
    .add_local_file(LOCAL_SCRIPT.as_posix(), REMOTE_SCRIPT, copy=True)
    .add_local_file(LOCAL_REDUCE.as_posix(), REMOTE_REDUCE, copy=True)
)

app = modal.App("v2-serve-lora-probe")


@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=60 * 60,
    volumes={MODEL_MOUNT: modal.Volume.from_name(MODEL_VOLUME)},
)
def probe() -> int:
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-u", REMOTE_SCRIPT], check=False)
    print(f"\n[exit] mig_06_serve_lora.py return code = {proc.returncode}", flush=True)
    return proc.returncode


@app.function(
    image=image,
    cpu=2.0,
    timeout=15 * 60,
    volumes={MODEL_MOUNT: modal.Volume.from_name(MODEL_VOLUME)},
)
def inspect() -> int:
    """开 GPU 之前先确认权重在、config 结构和探针假设的一致。"""
    import json

    model_dir = f"{MODEL_MOUNT}/qwen3-omni"
    print(f"===== {model_dir} =====", flush=True)
    if not os.path.isdir(model_dir):
        print(f"  目录不存在，卷根下有: {os.listdir(MODEL_MOUNT)}", flush=True)
        return 1
    entries = sorted(os.listdir(model_dir))
    print(f"  {len(entries)} 个条目，前 12 个: {entries[:12]}", flush=True)

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"\n  顶层键: {sorted(cfg)}", flush=True)
    thinker = cfg.get("thinker_config", cfg)
    print(f"  thinker_config 键: {sorted(thinker)}", flush=True)
    text = thinker.get("text_config", thinker)
    keep = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "num_hidden_layers")
    print(f"  text_config 形状: { {k: text.get(k) for k in keep} }", flush=True)

    print(f"\n  PYTHONPATH: {os.environ.get('PYTHONPATH', '')}", flush=True)
    import sglang

    print(f"  sglang {sglang.__version__} 来自 {os.path.dirname(sglang.__file__)}", flush=True)
    from sglang.srt.models import qwen3_omni_moe

    has_gate = hasattr(qwen3_omni_moe.Qwen3OmniMoeForConditionalGeneration, "should_apply_lora")
    print(f"  should_apply_lora 存在: {has_gate}", flush=True)
    return 0 if has_gate else 1


@app.function(image=image, cpu=2.0, timeout=15 * 60)
def reduce_ab() -> int:
    """探针六：CPU 张量的 reduce 越界 A/B，不需要 GPU。"""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-u", REMOTE_REDUCE], check=False)
    print(f"\n[exit] mig_07_cpu_reduce.py return code = {proc.returncode}", flush=True)
    return proc.returncode


@app.local_entrypoint()
def main(stage: str = "probe") -> None:
    fn = {"inspect": inspect, "reduce": reduce_ab, "probe": probe}[stage]
    rc = fn.remote()
    if rc != 0:
        raise SystemExit(f"{stage} 未通过（return={rc}）")
    print(f"\n{stage} 通过。", flush=True)
