"""探针四 —— sglang 侧 _lora_pattern 对真实模块名的命中（容器内运行）。

要回答的问题：
    我们给 Qwen3OmniMoeForConditionalGeneration 写的 _lora_pattern，
    对 sglang 真实建出来的模块树来说，命中的到底是哪些模块？
    以及：如果没有这个门（也就是上游现在的状态），会误挂到哪里？

省钱做法：
    不加载任何权重，把层数压到 2（文本 / 音频 / 视觉三个塔都压），
    专家数压到 4，只要模块树的**名字**是对的就够了。

判据：
    1. 门命中的模块必须全部落在 thinker 的文本主干里
    2. 门命中的模块里不能出现 audio_tower / visual
    3. 纯后缀匹配（上游当前行为）必须确实命中塔里的模块
       —— 这一条是反证，证明这个门不是可有可无的
    4. 加了门之后真正被包的模块数 == 2 * 文本层数（每层 qkv_proj + o_proj）

用法：python mig_05_sglang_lora_scope.py
"""

from __future__ import annotations

import os
import sys
import traceback

OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

# 训练侧 --lora-target-modules linear_qkv linear_proj 对应到 sglang 的名字。
TARGET_MODULES = {"qkv_proj", "o_proj"}

TEXT_LAYERS = 2
TOWER_LAYERS = 2
NUM_EXPERTS = 4

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _failures.append(tag)


def _shrink(cfg, names: list[str], value: int, label: str) -> None:
    """把 cfg 上第一个存在的字段压到 value。"""
    for n in names:
        if hasattr(cfg, n):
            old = getattr(cfg, n)
            setattr(cfg, n, value)
            print(f"    {label}: {n} {old} -> {value}", flush=True)
            return
    print(f"    {label}: 没找到 {names} 中的任何字段，保持原样", flush=True)


def setup_env() -> None:
    """把 sglang 建模型所需的全局状态立起来（单卡、TP=1）。"""
    import torch

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    server_args = ServerArgs(model_path=OMNI_CKPT)
    set_global_server_args_for_scheduler(server_args)

    torch.cuda.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29517",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    # LayerCommunicator 在建 decoder layer 时会问 attention DP 的大小，
    # 所以哪怕不开 DP attention 也得先把它初始化了。
    model_config = ModelConfig.from_server_args(server_args)
    initialize_dp_attention(server_args, model_config)
    print("    分布式环境就绪（world=1, TP=1, DP attention 已初始化）", flush=True)


def build_model():
    """用压过的 config 建 sglang 侧的 Omni，只为拿模块名。"""
    import torch
    from transformers import AutoConfig

    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    cfg = AutoConfig.from_pretrained(OMNI_CKPT, trust_remote_code=True)
    thinker = cfg.thinker_config

    print("  压缩配置：", flush=True)
    _shrink(thinker.text_config, ["num_hidden_layers"], TEXT_LAYERS, "文本层")
    _shrink(
        thinker.text_config,
        ["num_experts", "n_routed_experts", "num_local_experts"],
        NUM_EXPERTS,
        "专家数",
    )
    _shrink(
        thinker.audio_config,
        ["num_hidden_layers", "encoder_layers"],
        TOWER_LAYERS,
        "音频层",
    )
    _shrink(
        thinker.vision_config,
        ["depth", "num_hidden_layers"],
        TOWER_LAYERS,
        "视觉层",
    )

    with torch.device("meta"):
        model = Qwen3OmniMoeForConditionalGeneration(cfg)
    return model, thinker.text_config.num_hidden_layers


def main() -> None:
    print("========== 探针四：sglang 侧 LoRA 作用范围 ==========", flush=True)

    print("\n[1] 立起 sglang 全局环境", flush=True)
    try:
        setup_env()
    except Exception:  # noqa: BLE001
        _fail("env", "环境初始化失败：\n" + traceback.format_exc())
        sys.exit(1)

    print("\n[2] 建模型（meta device，不加载权重）", flush=True)
    try:
        model, n_text_layers = build_model()
    except Exception:  # noqa: BLE001
        _fail("build", "建模型失败：\n" + traceback.format_exc())
        sys.exit(1)

    names = [n for n, _ in model.named_modules() if n]
    print(f"    模块总数: {len(names)}", flush=True)

    print("\n[3] 三个集合", flush=True)
    gate_hits = [n for n in names if model.should_apply_lora(n)]
    suffix_hits = [n for n in names if n.split(".")[-1] in TARGET_MODULES]
    effective = [n for n in suffix_hits if model.should_apply_lora(n)]

    print(f"\n    门命中（should_apply_lora 为真）: {len(gate_hits)}", flush=True)
    for n in gate_hits:
        print(f"        {n}", flush=True)

    print(
        f"\n    纯后缀匹配命中（上游当前行为，target={sorted(TARGET_MODULES)}）: {len(suffix_hits)}",
        flush=True,
    )
    for n in suffix_hits:
        mark = "" if model.should_apply_lora(n) else "   <-- 会被门挡掉"
        print(f"        {n}{mark}", flush=True)

    print(f"\n    实际会被包上 LoRA 的（两者交集）: {len(effective)}", flush=True)
    for n in effective:
        print(f"        {n}", flush=True)

    print("\n[4] 判据", flush=True)

    tag = "gate-scope"
    bad = [n for n in gate_hits if not n.startswith("thinker.")]
    if bad:
        _fail(tag, f"门命中了 thinker 之外的模块: {bad}")
    else:
        _ok(tag, "门命中的模块全部在 thinker 下")

    tag = "no-tower"
    leaked = [n for n in gate_hits if "audio_tower" in n or "visual" in n]
    if leaked:
        _fail(tag, f"门放进了塔里的模块: {leaked}")
    else:
        _ok(tag, "门命中的模块里没有 audio_tower / visual")

    tag = "gate-necessary"
    tower_suffix = [n for n in suffix_hits if "audio_tower" in n or "visual" in n]
    if not tower_suffix:
        _fail(
            tag,
            "纯后缀匹配没有命中任何塔里的模块 —— 与探针一的结论矛盾，需要复查",
        )
    else:
        _ok(
            tag,
            f"纯后缀匹配命中了 {len(tower_suffix)} 个塔里的模块，"
            f"例如 {tower_suffix[0]} —— 证明这个门是必需的",
        )

    tag = "effective-count"
    expected = 2 * n_text_layers
    if len(effective) != expected:
        _fail(
            tag,
            f"实际包上的模块数为 {len(effective)}，预期 {expected}"
            f"（{n_text_layers} 层 x qkv_proj/o_proj）",
        )
    else:
        _ok(tag, f"实际包上 {len(effective)} 个模块 = {n_text_layers} 层 x 2，符合预期")

    print("\n========== 结论 ==========", flush=True)
    if _failures:
        print(f"  [PROBE X] 失败判据: {sorted(set(_failures))}", flush=True)
        sys.exit(1)
    print(
        "  [PROBE OK] _lora_pattern 只放行 thinker 文本主干的注意力投影；"
        "没有这个门，上游会把塔里的同名模块一并包上。",
        flush=True,
    )


if __name__ == "__main__":
    main()
