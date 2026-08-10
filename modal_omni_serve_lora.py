"""sglang-omni 的 LoRA 热加载「GPU 端到端」验证（S2S 前置：B1）。

目标：真的把 sglang-omni 的 **text-only thinker** 服务在 GPU 上拉起来，
    1. baseline 生成一段文本；
    2. HTTP POST /load_lora_adapter 热加载一个 LoRA adapter；
    3. 带 stage_params.thinker.lora_name 再生成，对比输出是否变化。
若输出变化 → 证明「sglang-omni 能热加载 LoRA 并生效」（这次加的接线在 GPU 上端到端通）。

分两个入口，先便宜后贵：
  - `modal run modal_omni_serve_lora.py::probe`  —— CPU，查版本 + 试导入启动链，几分钟。
  - `modal run modal_omni_serve_lora.py`          —— GPU(A100-80GB)，真拉服务 + 热加载对比。

镜像策略同 modal_test_sglang_omni：slime 预构建镜像（含 torch + sgl-kernel），
本地 sglang 0.5.12 fork + sglang-omni 源码用 PYTHONPATH 注入，缺的依赖再 pip 补。
"""

from __future__ import annotations

import os
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent
SGLANG_LOCAL = HERE / "sglang" / "python"
SGLANG_OMNI_LOCAL = HERE / "sglang-omni"

SGLANG_REMOTE = "/root/sglang_src/python"
SGLANG_OMNI_REMOTE = "/root/sglang-omni"
PYTHONPATH = f"{SGLANG_REMOTE}:{SGLANG_OMNI_REMOTE}"

MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_DIR = "/models/qwen3-omni"

# v1 冻结镜像 = slime nightly-dev-20260428a（:latest 会漂移，认定过程见 README_v1.md）
BASE_IMAGE = os.environ.get(
    "RELAX_BASE_IMAGE",
    "slimerl/slime@sha256:bd219aba21be6e404ff09e385f34f40993b60773b928e13f341e8d77590da6aa",
)

# sglang-omni 启动链需要的第三方依赖（尽量一次装齐，避免反复试错）。
# 用本地 sglang fork 覆盖，故不装 sglang；torch 用 slime 镜像自带（勿动）。
OMNI_DEPS = (
    "pip install --no-cache-dir "
    "typer pyzmq msgpack pydantic pyyaml xxhash httpx fastapi uvicorn pybase64 "
    "requests pillow accelerate safetensors soundfile librosa av qwen-vl-utils "
    "|| true"
)
# sglang-omni 的 Qwen3OmniMoeProcessor 需要 transformers 5.6（slime 自带 4.57.1，
# 其 processor 缺 image_token 等属性）。sglang-omni 本就把 sglang 0.5.12 + transformers
# 5.6 绑定为一套，我们的 sglang fork 也对齐 0.5.12，故一起升级。
TRANSFORMERS_DEP = "pip install --no-cache-dir 'transformers==5.6.0' || true"

image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .run_commands(OMNI_DEPS)
    .run_commands(TRANSFORMERS_DEP)
    .add_local_dir(
        SGLANG_LOCAL.as_posix(),
        SGLANG_REMOTE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        SGLANG_OMNI_LOCAL.as_posix(),
        SGLANG_OMNI_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/node_modules"],
    )
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App("sglang-omni-lora-e2e")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


@app.function(image=image, cpu=2.0, timeout=900)
def probe() -> None:
    """CPU 探针：查版本 + 试导入 sglang-omni 启动链（不加载模型、不起 GPU）。"""
    import importlib
    import subprocess
    import sys

    print("=" * 70)
    print("[probe] 版本自检")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch,transformers; print('torch', torch.__version__);"
            "print('transformers', transformers.__version__);"
            "import sglang; print('sglang', sglang.__file__)",
        ],
        cwd=SGLANG_OMNI_REMOTE,
    )

    print("=" * 70)
    print("[probe] 逐个导入启动链关键模块（报缺什么就补什么）")
    mods = [
        "sglang.srt.lora.lora_manager",
        "sglang.srt.models.qwen3_vl_moe",
        "sglang_omni.models.qwen3_omni.config",
        "sglang_omni.models.qwen3_omni.components.sglang_thinker",
        "sglang_omni.models.qwen3_omni.request_builders",
        "sglang_omni.scheduling.omni_scheduler",
        "sglang_omni.model_runner.model_worker",
        "sglang_omni.serve",
        "sglang_omni.cli.serve",
    ]
    failed = []
    for m in mods:
        code = (
            f"import importlib; importlib.import_module('{m}'); "
            f"print('  OK  {m}')"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=SGLANG_OMNI_REMOTE)
        if r.returncode != 0:
            failed.append(m)
            print(f"  FAIL {m}")

    print("=" * 70)
    if failed:
        print("[probe] 以下模块导入失败，需补依赖或修代码：")
        for m in failed:
            print("   -", m)
        raise SystemExit(1)

    # 能导入就试着构建 text-only config（不加载权重）
    print("[probe] 构建 Qwen3OmniPipelineConfig(text-only) ...")
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig;"
            "c=Qwen3OmniPipelineConfig(model_path='/models/qwen3-omni');"
            "print('stages:', [s.name for s in c.stages])",
        ],
        cwd=SGLANG_OMNI_REMOTE,
    )
    if r.returncode != 0:
        raise SystemExit("[probe] 构建 text-only config 失败")
    print("[probe] ALL OK —— 可以上 GPU 跑 e2e")


# LoRA 配置（与训练基线一致的最小集；rank 16）
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _serve_thinker() -> None:
    """在子进程里拉起 sglang-omni text-only thinker(TP4，带 LoRA)。

    必须是模块级函数，spawn 才能 pickle。复用 CLI 的 config 构建 helper
    （TP 放置逻辑已测过），再补 attention_backend=triton。
    """
    from sglang_omni.cli.serve import (
        _apply_stage_server_args_override,
        apply_parallelism_cli_overrides,
        apply_thinker_server_args_cli_overrides,
    )
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
    from sglang_omni.serve import launch_server

    config = Qwen3OmniPipelineConfig(model_path=MODEL_DIR)
    apply_parallelism_cli_overrides(
        config,
        thinker_tp_size=1,
        thinker_gpus="0",
        talker_gpu=None,
        code2wav_gpu=None,
    )
    apply_thinker_server_args_cli_overrides(
        config,
        cpu_offload_gb=None,
        quantization=None,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        lora_target_modules="qkv_proj,o_proj",
        max_loras_per_batch=1,
    )
    _apply_stage_server_args_override(
        config,
        stage_name="thinker",
        updates={
            "attention_backend": "triton",
            "disable_cuda_graph": True,
            "disable_custom_all_reduce": True,
            # TP=1 fits the ~57GB thinker on one 80GB GPU (weights 28.6GB/rank at
            # TP=2 ⇒ ~57GB total); bump the static fraction to leave room for KV +
            # encoders now that there is no second GPU to shard onto.
            "mem_fraction_static": 0.85,
            # Force the simplest synchronous scheduler loop (_event_loop_normal).
            # The overlap loop (enable_overlap default True) two-steps decode with
            # future_map and per-step overlap isolation; the smoke test does not
            # need overlap throughput.
            "disable_overlap_schedule": True,
        },
        reason="e2e Qwen3-Omni sglang settings",
    )
    launch_server(config, host="127.0.0.1", port=8000, model_name="qwen3-omni")


@app.function(
    image=image,
    gpu="A100-80GB:1",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def e2e() -> None:
    """GPU：拉起 text-only thinker(TP2) + enable_lora，热加载 adapter，对比生成。"""
    import multiprocessing
    import os
    import time

    import requests

    os.environ.setdefault("SGLANG_OMNI_STARTUP_TIMEOUT", "1800")
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    # 让 stage 子进程把崩溃 traceback 落盘（同容器共享 /tmp），规避平台日志只留尾部。
    stage_err_dir = "/tmp/sglang_stage_errors"
    os.environ["SGLANG_OMNI_STAGE_ERROR_DIR"] = stage_err_dir

    def _dump_stage_errors() -> None:
        import glob

        files = sorted(glob.glob(f"{stage_err_dir}/*.log"))
        if not files:
            print("[e2e] （无 stage 子进程 traceback 落盘）")
            return
        for fp in files:
            print("=" * 70)
            print(f"[e2e] STAGE ERROR {fp}:")
            try:
                with open(fp, encoding="utf-8") as fh:
                    print(fh.read())
            except Exception as exc:  # noqa: BLE001
                print(f"  <读取失败: {exc}>")

    assert os.path.exists(f"{MODEL_DIR}/config.json"), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json"
    )

    # ---- 1) 后台进程拉起 sglang-omni text-only thinker(TP4，带 LoRA)----
    # daemon=False：sglang-omni 会为每个 stage spawn 子进程，daemon 进程不允许有子进程。
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=_serve_thinker, daemon=False)
    p.start()

    base_url = "http://127.0.0.1:8000"

    def _healthy() -> bool:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    print("[e2e] 等待服务就绪（30B 首次加载可能数分钟）...")
    for i in range(240):
        if _healthy():
            print(f"[e2e] 服务就绪，用时约 {i * 5}s")
            break
        if not p.is_alive():
            _dump_stage_errors()
            raise SystemExit("[e2e] serve 进程已退出，启动失败")
        time.sleep(5)
    else:
        _dump_stage_errors()
        raise SystemExit("[e2e] 等待健康超时")

    prompt = "Translate to Chinese: The weather is nice today."

    def _gen(lora_name: str | None) -> tuple[str, list | None]:
        # sglang-omni 的 preprocessing 期望 chat messages（不是裸 prompt string）；
        # 裸 prompt 会在 normalize_messages() 抛 "expects a list of chat messages"。
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
            "stream": False,
            "return_logprob": True,
        }
        if lora_name:
            payload["stage_params"] = {"thinker": {"lora_name": lora_name}}
        # First (cold) forward JIT-compiles triton LoRA + MoE kernels with
        # disable_cuda_graph=True, so the very first generate can take minutes.
        r = requests.post(f"{base_url}/generate", json=payload, timeout=600)
        if r.status_code != 200:
            print(f"[e2e] /generate {r.status_code} body: {r.text[:800]}")
            # 给子进程一点时间把 traceback 落盘
            time.sleep(3)
            _dump_stage_errors()
            r.raise_for_status()
        body = r.json()
        return body.get("text", ""), (body.get("meta_info") or {}).get(
            "output_token_logprobs"
        )

    # ---- 2) baseline（无 LoRA）----
    base_out, base_logprobs = _gen(None)
    print(f"[e2e] BASE 输出: {base_out!r}")
    print(f"[e2e] BASE logprobs: {base_logprobs!r}")

    # ---- 3) 构造一个随机 PEFT adapter 落盘（lora_B 非零 → 输出会变）----
    adapter_dir = _build_random_peft_adapter()

    lr = requests.post(
        f"{base_url}/load_lora_adapter",
        json={
            "lora_name": "smoke",
            "lora_path": adapter_dir,
            "stages": ["thinker"],
        },
        timeout=300,
    )
    print(f"[e2e] load_lora_adapter -> {lr.status_code} {lr.text[:300]}")
    if lr.status_code != 200:
        raise SystemExit("[e2e] 热加载 LoRA 失败")

    # ---- 4) 带 LoRA 再生成 ----
    lora_out, lora_logprobs = _gen("smoke")
    print(f"[e2e] LORA 输出: {lora_out!r}")
    print(f"[e2e] LORA logprobs: {lora_logprobs!r}")

    print("=" * 70)
    if lora_out != base_out or lora_logprobs != base_logprobs:
        print("[e2e] PASS —— 带 LoRA 输出与 base 不同，热加载生效")
    else:
        print("[e2e] WARN —— 输出相同；可能 adapter 未生效（检查命名/路由）")


def _build_random_peft_adapter() -> str:
    """在 /tmp 写一个随机初始化的 PEFT LoRA adapter（thinker 语言体 attn）。"""
    import json
    import os

    import safetensors.torch
    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # Qwen3-Omni thinker 文本骨干维度（从 config 取，兜底常见值）
    tc = getattr(cfg, "thinker_config", None) or cfg
    text_cfg = getattr(tc, "text_config", None) or tc
    hidden = int(getattr(text_cfg, "hidden_size", 2048))
    n_heads = int(getattr(text_cfg, "num_attention_heads", 32))
    n_kv = int(getattr(text_cfg, "num_key_value_heads", 4))
    head_dim = int(getattr(text_cfg, "head_dim", hidden // n_heads))
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 48))
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    rank = LORA_RANK
    sd: dict[str, torch.Tensor] = {}
    # PEFT 命名：base_model.model.<module>.lora_A/B.weight
    # sglang-omni thinker 文本体层在 model.layers.*（无 thinker. 前缀）
    for i in range(n_layers):
        pre = f"base_model.model.model.layers.{i}.self_attn"
        for name, out_dim in [
            ("q_proj", q_dim),
            ("k_proj", kv_dim),
            ("v_proj", kv_dim),
            ("o_proj", hidden),
        ]:
            in_dim = q_dim if name == "o_proj" else hidden
            sd[f"{pre}.{name}.lora_A.weight"] = torch.randn(rank, in_dim) * 0.02
            # lora_B 非零，确保输出可见变化
            sd[f"{pre}.{name}.lora_B.weight"] = torch.randn(out_dim, rank) * 0.02

    adapter_dir = "/tmp/smoke_adapter"
    os.makedirs(adapter_dir, exist_ok=True)
    safetensors.torch.save_file(sd, f"{adapter_dir}/adapter_model.safetensors")
    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": MODEL_DIR,
        "r": rank,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": 0.0,
        "target_modules": LORA_TARGET,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(f"{adapter_dir}/adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"[e2e] 已写随机 PEFT adapter -> {adapter_dir} ({len(sd)} tensors)")
    return adapter_dir


@app.local_entrypoint()
def main() -> None:
    e2e.remote()


@app.local_entrypoint()
def probe_only() -> None:
    probe.remote()
