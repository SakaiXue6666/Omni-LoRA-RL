"""在 Modal 上跑 sglang 的 LoRA × RL 系列实验（E–J）。

为什么这么搭环境：
  你本地的 sglang 是 `main` 分支 + 你自己加的 LoRA 改动，比 v0.5.9 镜像新得多
  （所以 v0.5.9 里没有 FusedMoEWithLoRA / virtual_experts 这些新符号）。
  因此基础镜像用 `lmsysorg/sglang:latest`（从 main 构建，二进制依赖匹配），
  再把【整份】本地 python 源码通过 PYTHONPATH 覆盖进去 —— 不逐文件打 patch、
  也不需要为版本错配写 stub。

各实验入口（见文件底部 @app.local_entrypoint）：
  实验 E（默认）         ：启动时静态注入 --lora-paths，验证最基础的挂载路径
  实验 F（--dynamic 1）  ：起 server 后用 /load_lora_adapter 动态加载（旧版会崩的那种）
  实验 G（::rl）         ：in-process Engine 模拟 RL 热更新循环（生效/可逆/稳定）
  实验 H（::flat）       ：flattened_bucket 快速同步路径 vs 普通逐 tensor 路径
  实验 I（::mega）       ：Megatron fused-qkv 与 PEFT split-qkv 是否数值等价
  实验 J（::relax）      ：完整复刻 Relax 的 HTTP + CUDA IPC 同步链路

用法（在 d:\\Li_Lab\\RL 下运行）：
    pip install modal
    modal token new                         # 首次授权
    modal run modal_run.py                  # 实验 E（静态注入）
    modal run modal_run.py --dynamic 1      # 实验 F（动态加载）
    modal run modal_run.py --alpha 0.001    # alpha 极小的对照实验
    modal run modal_run.py::rl              # 实验 G
    modal run modal_run.py::flat            # 实验 H
    modal run modal_run.py::mega            # 实验 I
    modal run modal_run.py::relax           # 实验 J

注意：Windows 控制台要先切 UTF-8，否则 modal 打印 ✓ 会因 gbk 编码报错：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
"""

from __future__ import annotations

import os
import pathlib

import modal

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
LOCAL_SGLANG_PYTHON = HERE / "sglang" / "python"   # 本地完整源码（main 分支）

# 本地源码在镜像里的落点；放在 PYTHONPATH 最前，让【整份】本地代码
# （layers.py、lora_manager.py …）作为一个一致的整体被 import。
# 编译依赖（sgl-kernel、flashinfer）仍由镜像提供。
SRC_DIR = "/root/sglang_src/python"

MODEL_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_DIR = "/models/qwen3-omni"          # 存在缓存 Volume 上
LORA_DIR = "/root/toy-lora-thinker"        # 每次运行重建（很便宜）
SERVER_PORT = 30000

# ---------------------------------------------------------------------------
# 镜像：本地源码是 sglang `main`（需要 CUDA 13 / sgl-kernel 0.4.x / flashinfer
# 0.6.11），所以基础镜像用 `latest`（从 main 构建），再把整份本地 python 源码
# 通过 PYTHONPATH 覆盖进去。
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("lmsysorg/sglang:latest", add_python=None)
    .pip_install("huggingface_hub[hf_transfer]", "safetensors")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": SRC_DIR})
    .add_local_dir(
        LOCAL_SGLANG_PYTHON.as_posix(),
        SRC_DIR,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App("sglang-omni-lora")
model_volume = modal.Volume.from_name("qwen3-omni-weights", create_if_missing=True)


# ---------------------------------------------------------------------------
# 在容器内运行的辅助函数
# ---------------------------------------------------------------------------
def _verify_local_source() -> None:
    """确认 `import sglang` 解析到我们覆盖进去的本地源码，且 LoRA 相关改动
    （should_apply_lora 闸门 + FusedMoEWithLoRA）都在位、彼此一致。"""
    import sglang

    base = os.path.dirname(sglang.__file__)
    assert base.startswith(SRC_DIR), (
        f"sglang 解析到 {base}，本应在 {SRC_DIR} 下。"
        "PYTHONPATH 覆盖没生效。"
    )
    print(f"[src] sglang 来自本地源码: {base}")

    mgr_path = os.path.join(base, "srt", "lora", "lora_manager.py")
    mgr = open(mgr_path, encoding="utf-8").read()
    assert 'getattr(self.base_model, "should_apply_lora", None)' in mgr, "闸门代码缺失!"
    print("[src] should_apply_lora 闸门已确认存在。")

    # 当年在 v0.5.9 上因为缺这个符号而失败，这里必须存在
    from sglang.srt.lora.layers import FusedMoEWithLoRA  # noqa: F401

    print("[src] FusedMoEWithLoRA import 成功（整份源码一致）。")


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        print(f"[model] 已缓存在 {MODEL_DIR}，跳过下载。")
        return
    print(f"[model] 正在下载 {MODEL_REPO} -> {MODEL_DIR}（可能要一会儿）...")
    snapshot_download(
        MODEL_REPO,
        local_dir=MODEL_DIR,
        max_workers=8,
    )
    model_volume.commit()
    print("[model] 下载完成并已提交到 volume。")


RANK = 32
HIDDEN = 2048
# Qwen3-Omni thinker 上每个注意力投影的 out_features
ATTN_DIMS = {"q_proj": 4096, "k_proj": 512, "v_proj": 512, "o_proj": 2048}
ATTN_IN = {"q_proj": HIDDEN, "k_proj": HIDDEN, "v_proj": HIDDEN, "o_proj": 4096}
NUM_LAYERS = 48


def _lora_config_dict(alpha: float = 32.0) -> dict:
    return {
        "base_model_name_or_path": MODEL_DIR,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": RANK,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def _make_lora_tensors(b_scale: float, a_scale: float = 0.02, seed: int | None = None):
    """为 thinker 的注意力投影构造一份内存中的 PEFT LoRA 权重字典。

    b_scale == 0  -> lora_B 全 0 -> B@A == 0 -> 数学上的 no-op（不改变输出）。
    b_scale  > 0  -> 非零增量，相当于一个"训练过"的权重版本。
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    tensors: dict[str, "torch.Tensor"] = {}
    for layer in range(NUM_LAYERS):
        for mod, out_f in ATTN_DIMS.items():
            prefix = f"base_model.model.thinker.model.layers.{layer}.self_attn.{mod}"
            a = torch.randn(RANK, ATTN_IN[mod], dtype=torch.bfloat16) * a_scale
            if b_scale == 0:
                b = torch.zeros(out_f, RANK, dtype=torch.bfloat16)
            else:
                b = torch.randn(out_f, RANK, dtype=torch.bfloat16) * b_scale
            tensors[f"{prefix}.lora_A.weight"] = a
            tensors[f"{prefix}.lora_B.weight"] = b
    return tensors


def _make_toy_lora(alpha: float = 32.0) -> None:
    """把一个 rank-32、B=0 的 adapter 写到磁盘（静态 --lora-paths 实验用）。"""
    import json

    from safetensors.torch import save_file

    os.makedirs(LORA_DIR, exist_ok=True)
    tensors = _make_lora_tensors(b_scale=0.0)
    save_file(tensors, os.path.join(LORA_DIR, "adapter_model.safetensors"))
    with open(os.path.join(LORA_DIR, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(_lora_config_dict(alpha), f, indent=2)
    print(f"[lora] wrote {len(tensors)} tensors (alpha={alpha}) to {LORA_DIR}")


def _wait_until_ready(timeout: float = 900.0) -> None:
    import time
    import urllib.request

    url = f"http://127.0.0.1:{SERVER_PORT}/get_model_info"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    print(f"[server] ready after {time.time() - start:.0f}s")
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server 没能在规定时间内就绪")


def _chat_prompt(user_msg: str) -> str:
    """把用户消息套进 Qwen3 的 chat 模板，让这个 instruct 模型真正去回答，
    而不是一上来就吐 <|im_end|>（空输出）。"""
    return (
        "<|im_start|>user\n"
        f"{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _generate(user_msg: str, lora_path: str | None = None) -> dict:
    import json
    import urllib.request

    payload = {
        "text": _chat_prompt(user_msg),
        "sampling_params": {"max_new_tokens": 50, "temperature": 0},
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _load_lora_dynamic(lora_name: str, lora_path: str) -> None:
    """调用 /load_lora_adapter，在 server 启动后动态注入 adapter。"""
    import json
    import urllib.request

    payload = {"lora_name": lora_name, "lora_path": lora_path}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/load_lora_adapter",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    if not resp.get("success"):
        raise RuntimeError(f"/load_lora_adapter 失败: {resp}")
    print(f"[lora] 动态加载成功: {resp}")


# ---------------------------------------------------------------------------
# 实验 E / F 的 Modal 入口
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def run_experiment(alpha: float = 32.0, dynamic: bool = False) -> None:
    import subprocess
    import sys

    mode = "F (动态 /load_lora_adapter)" if dynamic else "E (静态 --lora-paths)"
    print(f"\n========== 实验 {mode} ==========")

    _verify_local_source()
    _download_model()
    _make_toy_lora(alpha=alpha)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_DIR,
        "--enable-lora",
        "--max-lora-rank", "32",
        "--max-loras-per-batch", "2",
        "--lora-target-modules", "qkv_proj", "o_proj",
        "--tp-size", "1",
        "--disable-cuda-graph",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--mem-fraction-static", "0.85",
        "--log-level", "info",
    ]
    # 实验 E：静态注入，server 启动时就知道有这个 adapter
    # 实验 F：动态加载，server 启动后再调 /load_lora_adapter
    if not dynamic:
        cmd += ["--lora-paths", f"toy-thinker={LORA_DIR}"]

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    print("[server] 启动命令:\n  " + " ".join(cmd))
    server = subprocess.Popen(cmd, env=env)

    try:
        _wait_until_ready()

        if dynamic:
            _load_lora_dynamic("toy-thinker", LORA_DIR)

        prompt = "Hello, who are you?"
        base = _generate(prompt)
        lora = _generate(prompt, lora_path="toy-thinker")

        print("\n================ 结果 ================")
        print(f"模式: 实验 {mode}")
        print("BASE（不带 LoRA）:")
        print("  文本  :", repr(base["text"][:200]))
        print("  tokens:", base["meta_info"]["completion_tokens"])
        print("LORA (toy-thinker, B=0):")
        print("  文本  :", repr(lora["text"][:200]))
        print("  tokens:", lora["meta_info"]["completion_tokens"])
        print("=====================================")

        base_ok = base["meta_info"]["completion_tokens"] > 2
        lora_ok = lora["meta_info"]["completion_tokens"] > 2

        if not base_ok:
            print("结论: BASE 自身就坏了（token 太少）—— 环境问题。")
        elif base["text"] == lora["text"]:
            print("结论: 两者完全一致 -> 这条路径是干净的。")
            if dynamic:
                print("       => main 已经修好了动态加载的 bug。")
                print("       => 找出修复 commit: git log --oneline | grep -i lora")
            else:
                print("       =>（预期内）bug 在动态 /load_lora_adapter 路径上。")
        else:
            lora_tokens = lora["meta_info"]["completion_tokens"]
            print(f"结论: 不一致（lora tokens={lora_tokens}）-> 这条路径会破坏输出。")
            if dynamic:
                print("       => main 仍存在动态加载 bug。")
                print("       => 去修 /load_lora_adapter 的 batch_info / permutation 刷新。")
            else:
                print("       => 意外：静态启动也坏了，bug 更深层。")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


@app.local_entrypoint()
def main(alpha: float = 32.0, dynamic: int = 0) -> None:
    """传 --dynamic 1 跑实验 F（动态 /load_lora_adapter）。"""
    run_experiment.remote(alpha=alpha, dynamic=bool(dynamic))


# ---------------------------------------------------------------------------
# 实验 G：RL 运行时 LoRA 热更新循环（进程内 Engine）
#
# 复刻 RL rollout worker 用 sglang 的方式：保持一个 engine 常驻，反复通过
# load_lora_adapter_from_tensors 从内存推入新的 LoRA 权重（不落盘）。
# 验证三个性质：
#   1. 生效（EFFECT）    —— 非零权重版本会让输出相对 base 改变。
#   2. 可逆（REVERSIBLE）—— 推入 B=0 版本后输出回到 base（说明旧权重没残留）。
#   3. 稳定（STABLE）    —— 连续 N 次更新不崩、不 OOM，显存有界。
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def run_rl_loop(steps: int = 8, max_loaded: int = 2) -> None:
    import torch

    import sglang as sgl

    print("\n========== 实验 G（RL 运行时 LoRA 热更新）==========")
    _verify_local_source()
    _download_model()

    engine = sgl.Engine(
        model_path=MODEL_DIR,
        enable_lora=True,
        max_lora_rank=RANK,
        lora_target_modules=["qkv_proj", "o_proj"],
        max_loras_per_batch=2,
        max_loaded_loras=max_loaded,
        tp_size=1,
        disable_cuda_graph=True,
        mem_fraction_static=0.85,
        log_level="info",
    )

    sampling = {"max_new_tokens": 50, "temperature": 0.0}
    prompt = _chat_prompt("Hello, who are you?")
    config = _lora_config_dict()

    def gen(lora_name: str | None = None) -> tuple[str, int]:
        kwargs = {"prompt": [prompt], "sampling_params": sampling}
        if lora_name is not None:
            kwargs["lora_path"] = [lora_name]
        out = engine.generate(**kwargs)[0]
        return out["text"], out["meta_info"]["completion_tokens"]

    def load(lora_name: str, b_scale: float, seed: int | None = None) -> None:
        tensors = _make_lora_tensors(b_scale=b_scale, seed=seed)
        res = engine.load_lora_adapter_from_tensors(
            lora_name=lora_name, tensors=tensors, config_dict=config
        )
        if not res.success:
            raise RuntimeError(f"加载 '{lora_name}' 失败: {res.error_message}")

    try:
        base_text, base_tok = gen()
        print(f"[G] BASE: tokens={base_tok} text={base_text[:120]!r}")

        # 1. 可逆性：B=0 版本必须精确复现 base
        load("zero-v", b_scale=0.0)
        z_text, z_tok = gen("zero-v")
        reversible = z_text == base_text
        print(f"[G] ZERO (B=0): tokens={z_tok} same_as_base={reversible}")

        # 2. 生效性：非零版本必须改变输出
        load("eff-v", b_scale=0.15, seed=123)
        e_text, e_tok = gen("eff-v")
        effect = e_text != base_text
        print(f"[G] NONZERO: tokens={e_tok} differs_from_base={effect} text={e_text[:120]!r}")

        # 3. 稳定性：连续 N 次热更新（每步都是全新权重版本）
        torch.cuda.reset_peak_memory_stats()
        crashed = False
        for i in range(steps):
            try:
                load(f"policy-v{i}", b_scale=0.1, seed=1000 + i)
                t, tok = gen(f"policy-v{i}")
                print(f"[G] step {i:>2}: tokens={tok} text={t[:60]!r}")
            except Exception as exc:  # noqa: BLE001
                crashed = True
                print(f"[G] step {i:>2}: 崩溃 -> {exc}")
                break
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        print("\n================ 结果（实验 G）================")
        print(f"  可逆 (B=0 == base)       : {reversible}")
        print(f"  生效 (非零 != base)      : {effect}")
        print(f"  稳定 ({steps} 次更新)         : {not crashed}")
        print(f"  循环中显存峰值           : {peak_gb:.1f} GB")
        print("==============================================")
        if reversible and effect and not crashed:
            print("结论: RL 热更新路径可用，可以接入 RL 框架了。")
        else:
            print("结论: RL 热更新路径有问题（看上面的标志位）。")
    finally:
        engine.shutdown()


@app.local_entrypoint()
def rl(steps: int = 8, max_loaded: int = 2) -> None:
    """跑实验 G:  modal run modal_run.py::rl"""
    run_rl_loop.remote(steps=steps, max_loaded=max_loaded)


# ---------------------------------------------------------------------------
# 实验 H：验证 FlattenedTensorBucket 权重同步路径
#
# 这正是 RL 框架（slime/verl/Miles）推 LoRA 增量时用的传输方式：把一堆带名字的
# tensor 打平进一个 bucket + metadata，一次性序列化，再用
# load_format="flattened_bucket" 加载。我们验证它产出的输出与普通逐 tensor
# 路径【完全一致】—— 也就是这条快速路径不引入任何漂移。
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def run_flattened() -> None:
    import sglang as sgl
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    print("\n========== 实验 H（flattened_bucket RL 权重同步路径）==========")
    _verify_local_source()
    _download_model()

    engine = sgl.Engine(
        model_path=MODEL_DIR,
        enable_lora=True,
        max_lora_rank=RANK,
        lora_target_modules=["qkv_proj", "o_proj"],
        max_loras_per_batch=2,
        max_loaded_loras=4,
        tp_size=1,
        disable_cuda_graph=True,
        mem_fraction_static=0.85,
        log_level="info",
    )

    sampling = {"max_new_tokens": 50, "temperature": 0.0}
    prompt = _chat_prompt("Hello, who are you?")
    config = _lora_config_dict()

    def gen(lora_name: str | None = None) -> str:
        kwargs = {"prompt": [prompt], "sampling_params": sampling}
        if lora_name is not None:
            kwargs["lora_path"] = [lora_name]
        return engine.generate(**kwargs)[0]["text"]

    try:
        # 同一份权重，用两种方式加载
        tensors = _make_lora_tensors(b_scale=0.15, seed=777)

        # 路径 1：普通逐 tensor
        r1 = engine.load_lora_adapter_from_tensors(
            lora_name="plain", tensors=tensors, config_dict=config
        )
        assert r1.success, f"plain 加载失败: {r1.error_message}"
        plain_text = gen("plain")

        # 路径 2：FlattenedTensorBucket（RL 同步路径）
        named = list(tensors.items())
        bucket = FlattenedTensorBucket(named_tensors=[(n, t) for n, t in named])
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)
        r2 = engine.load_lora_adapter_from_tensors(
            lora_name="flat",
            tensors=serialized,
            config_dict=config,
            load_format="flattened_bucket",
        )
        assert r2.success, f"flattened 加载失败: {r2.error_message}"
        flat_text = gen("flat")

        match = plain_text == flat_text

        print("\n================ 结果（实验 H）================")
        print(f"  普通逐 tensor : {plain_text[:120]!r}")
        print(f"  flattened     : {flat_text[:120]!r}")
        print(f"  完全一致       : {match}")
        print("==============================================")
        if match:
            print("结论: flattened_bucket 路径与普通路径一致，RL 传输方式可靠。")
        else:
            print("结论: 不一致 -> flattened_bucket 路径引入了漂移，需要排查。")
    finally:
        engine.shutdown()


@app.local_entrypoint()
def flat() -> None:
    """跑实验 H:  modal run modal_run.py::flat"""
    run_flattened.remote()


# ---------------------------------------------------------------------------
# 实验 I：Megatron 的 fused-qkv 约定 == PEFT 的 split-qkv 约定
#
# Megatron(-Bridge) 把注意力保存为单个 fused linear_qkv：一个【共享】的 lora_A
# 加一个 fused 的 lora_B。sglang 的 normalize_qkv_proj 能直接吃（内部把共享 A
# 复制 3 份）。我们证明：用同一份底层权重构造的 fused-qkv adapter 与 split q/k/v
# adapter 产出【完全相同】的输出 -> slime 可以直接推 Megatron 的 fused qkv，
# 不用拆开。
# ---------------------------------------------------------------------------
Q_OUT, KV_OUT = 4096, 512
QKV_OUT = Q_OUT + 2 * KV_OUT  # 5120


def _make_fused_and_split(b_scale: float = 0.12, seed: int = 2024):
    """从同一份随机底料派生出 (fused_tensors, split_tensors)，外加各自的 config
    target_modules。两者在数学上是完全等价的 adapter。"""
    import torch

    torch.manual_seed(seed)
    fused: dict[str, "torch.Tensor"] = {}
    split: dict[str, "torch.Tensor"] = {}
    for layer in range(NUM_LAYERS):
        p = f"base_model.model.thinker.model.layers.{layer}.self_attn"

        # ---- 注意力 qkv ----
        a_qkv = torch.randn(RANK, HIDDEN, dtype=torch.bfloat16) * 0.02  # 共享 A
        b_qkv = torch.randn(QKV_OUT, RANK, dtype=torch.bfloat16) * b_scale  # fused B

        # Megatron / fused 布局
        fused[f"{p}.qkv_proj.lora_A.weight"] = a_qkv
        fused[f"{p}.qkv_proj.lora_B.weight"] = b_qkv

        # PEFT / split 布局（同一个共享 A，B 按输出维度切片）
        for mod, lo, hi in (
            ("q_proj", 0, Q_OUT),
            ("k_proj", Q_OUT, Q_OUT + KV_OUT),
            ("v_proj", Q_OUT + KV_OUT, QKV_OUT),
        ):
            split[f"{p}.{mod}.lora_A.weight"] = a_qkv.clone()
            split[f"{p}.{mod}.lora_B.weight"] = b_qkv[lo:hi].clone()

        # ---- 输出投影 o_proj（两边相同）----
        a_o = torch.randn(RANK, Q_OUT, dtype=torch.bfloat16) * 0.02
        b_o = torch.randn(HIDDEN, RANK, dtype=torch.bfloat16) * b_scale
        fused[f"{p}.o_proj.lora_A.weight"] = a_o
        fused[f"{p}.o_proj.lora_B.weight"] = b_o
        split[f"{p}.o_proj.lora_A.weight"] = a_o.clone()
        split[f"{p}.o_proj.lora_B.weight"] = b_o.clone()

    return fused, split


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def run_megatron_contract() -> None:
    import sglang as sgl

    print("\n========== 实验 I（Megatron fused-qkv == PEFT split-qkv）==========")
    _verify_local_source()
    _download_model()

    engine = sgl.Engine(
        model_path=MODEL_DIR,
        enable_lora=True,
        max_lora_rank=RANK,
        lora_target_modules=["qkv_proj", "o_proj"],
        max_loras_per_batch=2,
        max_loaded_loras=4,
        tp_size=1,
        disable_cuda_graph=True,
        mem_fraction_static=0.85,
        log_level="info",
    )

    # return_logprob 提供一个确定性的数值指纹；即便（随机权重导致）输出退化成
    # 立即停止，这个指纹也仍然有意义、可比。
    sampling = {"max_new_tokens": 8, "temperature": 0.0}
    prompt = _chat_prompt("Hello, who are you?")

    def gen(name: str | None):
        kwargs = {
            "prompt": [prompt],
            "sampling_params": sampling,
            "return_logprob": True,
        }
        if name is not None:
            kwargs["lora_path"] = [name]
        out = engine.generate(**kwargs)[0]
        text = out["text"]
        # output_token_logprobs: 每项是 [logprob, token_id, token_text]
        olp = out["meta_info"].get("output_token_logprobs") or []
        ids = [int(t[1]) for t in olp]
        lps = [float(t[0]) for t in olp]
        return text, ids, lps

    def lp_close(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        return max((abs(a[i] - b[i]) for i in range(n)), default=0.0)

    try:
        # 用很小的扰动，让结果不至于退化（避免空输出）
        fused, split = _make_fused_and_split(b_scale=0.02)

        cfg_fused = _lora_config_dict()
        cfg_fused["target_modules"] = ["qkv_proj", "o_proj"]
        cfg_split = _lora_config_dict()
        cfg_split["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]

        rf = engine.load_lora_adapter_from_tensors(
            lora_name="fused", tensors=fused, config_dict=cfg_fused
        )
        assert rf.success, f"fused 加载失败: {rf.error_message}"
        f_text, f_ids, f_lps = gen("fused")

        rs = engine.load_lora_adapter_from_tensors(
            lora_name="split", tensors=split, config_dict=cfg_split
        )
        assert rs.success, f"split 加载失败: {rs.error_message}"
        s_text, s_ids, s_lps = gen("split")

        ids_match = f_ids == s_ids
        max_lp_diff = lp_close(f_lps, s_lps)
        numeric_match = ids_match and max_lp_diff < 1e-3

        print("\n================ 结果（实验 I）================")
        print(f"  fused (Megatron) 文本 : {f_text[:80]!r}")
        print(f"  split (PEFT)     文本 : {s_text[:80]!r}")
        print(f"  fused token ids       : {f_ids}")
        print(f"  split token ids       : {s_ids}")
        print(f"  token ids 是否一致    : {ids_match}")
        print(f"  logprob 最大绝对误差  : {max_lp_diff:.3e}")
        print("==============================================")
        if numeric_match:
            print("结论: fused-qkv == split-qkv（logprob 吻合）。slime 可以把 Megatron 的")
            print("       fused linear_qkv 直接当作 qkv_proj.lora_A/B 推过来。")
        else:
            print("结论: 不一致 -> fused 路径与 split 不同。需排查")
            print("       normalize_qkv_proj 的 fused 分支 / B 的切片顺序。")
    finally:
        engine.shutdown()


@app.local_entrypoint()
def mega() -> None:
    """跑实验 I:  modal run modal_run.py::mega"""
    run_megatron_contract.remote()


# ---------------------------------------------------------------------------
# 实验 J：验证 Relax 真实使用的那条权重同步链路
#
# Relax 是跟一个 SGLang【HTTP server】通信的：把经 MultiprocessingSerializer
# 序列化的 FlattenedTensorBucket（走 CUDA IPC）POST 到
# /load_lora_adapter_from_tensors —— 而不是 G/H/I 用的进程内 Engine。
# verl 当年的 bug #4065（CPU tensor 序列化问题）就出在这条路上。
# 我们逐字节复刻 Relax.SGLangEngine.load_lora_adapter_from_tensors：
#   - server 用 --enable-lora 启动（不带 --lora-paths）
#   - GPU 上的 tensor -> FlattenedTensorBucket -> 序列化 -> HTTP POST
#   - 通过 HTTP 带 lora_path 生成
# ---------------------------------------------------------------------------
def _http_post(endpoint: str, payload: dict) -> dict:
    import json
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def run_relax_http() -> None:
    import subprocess
    import sys

    import torch
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    print("\n========== 实验 J（Relax 风格 HTTP + IPC LoRA 同步）==========")
    _verify_local_source()
    _download_model()

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", MODEL_DIR,
        "--enable-lora",
        "--max-lora-rank", "32",
        "--max-loras-per-batch", "2",
        "--max-loaded-loras", "2",
        "--lora-target-modules", "qkv_proj", "o_proj",
        "--tp-size", "1",
        "--disable-cuda-graph",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--mem-fraction-static", "0.80",
        "--log-level", "info",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    print("[server] 启动中（HTTP，--enable-lora，不带 --lora-paths）")
    server = subprocess.Popen(cmd, env=env)

    try:
        _wait_until_ready()

        # base 输出（不挂 adapter）
        base = _generate("Hello, who are you?")
        base_text = base["text"]
        print(f"[J] BASE: tokens={base['meta_info']['completion_tokens']} {base_text[:100]!r}")

        # 在 GPU 上构造 Megatron 风格的 fused-qkv LoRA tensor（RL 里权重就在 GPU 上）
        fused, _ = _make_fused_and_split(b_scale=0.02)
        fused = {k: v.cuda() for k, v in fused.items()}
        config = _lora_config_dict()
        config["target_modules"] = ["qkv_proj", "o_proj"]

        # 完全按 Relax 的方式序列化：FlattenedTensorBucket -> MP 序列化器
        named = list(fused.items())
        bucket = FlattenedTensorBucket(named_tensors=named)
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        # POST 到 Relax.SGLangEngine.load_lora_adapter_from_tensors 调用的同一个端点
        resp = _http_post(
            "load_lora_adapter_from_tensors",
            {
                "lora_name": "rl-policy",
                "serialized_tensors": serialized,
                "config_dict": config,
                "load_format": "flattened_bucket",
                "pinned": False,
            },
        )
        load_ok = bool(resp.get("success"))
        print(f"[J] HTTP load_lora_adapter_from_tensors: success={load_ok} resp={resp}")

        lora = _generate("Hello, who are you?", lora_path="rl-policy")
        lora_text = lora["text"]
        lora_tok = lora["meta_info"]["completion_tokens"]
        print(f"[J] LORA: tokens={lora_tok} {lora_text[:100]!r}")

        coherent = lora_tok > 2
        print("\n================ 结果（实验 J）================")
        print(f"  HTTP IPC 加载成功      : {load_ok}")
        print(f"  lora 能正常生成 (>2tok): {coherent}")
        print(f"  base : {base_text[:80]!r}")
        print(f"  lora : {lora_text[:80]!r}")
        print("==============================================")
        if load_ok and coherent:
            print("结论: Relax 风格的 HTTP+IPC LoRA 同步端到端跑通。")
            print("       => Block 1（relax sglang_engine）的传输链路已验证。")
        else:
            print("结论: Relax 风格同步失败（看标志位）。接 Megatron 前先排查。")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


@app.local_entrypoint()
def relax() -> None:
    """跑实验 J:  modal run modal_run.py::relax"""
    run_relax_http.remote()
