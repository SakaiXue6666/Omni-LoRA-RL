"""排除风险：验证 Megatron-Bridge PEFT 的 LoRA 能正确挂到 Qwen3-Omni thinker。

这个脚本【不训练、不加载真实权重】，只用真实的 Megatron 模块构造一个最小模型树，
然后把 megatron.bridge.peft.LoRA 挂上去，逐条核对前面计划依赖的几个假设：

  1. adapter 参数命名 == `...<base>.adapter.linear_in/out.weight`
     （Block 3 的转换器就按这个名字抽权重）
  2. 普通 LoRA 会冻结所有基座参数，只有 adapter 可训
     （证明不需要 VLMLoRA，也不依赖 LLaVA 风格的属性名）
  3. 裸 target=["linear_qkv"] 会【过匹配】——把 LoRA 误挂到音频塔
     （这就是“坑②”，sglang 侧没有对应模块，必须避免）
  4. 用通配符把 target 限定到 language_model 后，音频塔不再被挂

运行环境：Relax 的 Megatron 训练镜像里（有 megatron.core / megatron.bridge /
transformer_engine）。单卡 TP=1 即可，甚至不需要真实 checkpoint。

    torchrun --nproc_per_node=1 verify_lora_attach.py

如果四项断言全过 -> Block 4 的“用普通 LoRA + 通配符 target”方案被钉死，可以放心写。
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def _init_megatron_tp1() -> None:
    """单进程初始化 Megatron 的并行状态（TP=PP=1）。"""
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # Megatron 初始化权重时需要 model-parallel RNG tracker，否则报
    # "cuda rng state model-parallel-rng is not added"
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)


def _build_attention_block() -> nn.Module:
    """构造一个真实的 Megatron self-attention（含 linear_qkv / linear_proj）。

    用真实模块而不是 nn.Linear，确保 PEFT 的类型判断（ColumnParallelLinear /
    TE*）走的是和真实 thinker 一样的分支。
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=4,          # GQA，制造 fused-qkv 的 q/k/v 维度差异
        ffn_hidden_size=512,
        kv_channels=32,
        use_cpu_initialization=not torch.cuda.is_available(),
        bf16=False,
        fp16=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        add_bias_linear=False,
    )
    # 优先用 TE spec（和真实 thinker 一致）；T4 等环境若 TE 构造异常，回退到本地 spec
    try:
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
        print("[build] 使用 TransformerEngine spec")
    except Exception as exc:  # noqa: BLE001
        print(f"[build] TE spec 失败({exc})，回退到 local spec")
        spec = get_gpt_layer_local_spec(qk_layernorm=True)
        layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
    layer.config = config
    return layer


def _build_omni_like_tree() -> nn.Module:
    """拼出一个近似 Qwen3-Omni 的结构：thinker.{language_model, audio_model}。

    audio_model 也用一个真实的 Megatron attention 块（同样带 self_attention.linear_qkv），
    这样它被过匹配挂上 LoRA 时，命名和 language_model 一致（都是 .adapter.linear_in/out），
    用来如实演示“坑②：裸 target 会误挂到音频塔”。
    """
    root = nn.Module()
    root.thinker = nn.Module()

    # thinker.language_model.decoder.layers.0.self_attention.{linear_qkv,linear_proj}
    root.thinker.language_model = nn.Module()
    root.thinker.language_model.decoder = nn.Module()
    root.thinker.language_model.decoder.layers = nn.ModuleList([_build_attention_block()])

    # thinker.audio_model.self_attention.linear_qkv  <- 不该被挂 LoRA
    root.thinker.audio_model = _build_attention_block()
    return root


def _is_adapter_name(n: str) -> bool:
    # 真实 Megatron 模块 -> .adapter.linear_in/out；纯 nn.Linear -> .lora_a/.lora_b
    return ".adapter." in n or ".lora_a" in n or ".lora_b" in n


def _adapter_param_names(model: nn.Module) -> list[str]:
    return [n for n, _ in model.named_parameters() if _is_adapter_name(n)]


def _trainable_param_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def _import_lora():
    """兼容不同 Megatron-Bridge 版本的 LoRA import 路径。"""
    try:
        from megatron.bridge.peft.lora import LoRA

        return LoRA
    except Exception:  # noqa: BLE001
        from megatron.bridge.peft import LoRA  # type: ignore

        return LoRA


def main() -> None:
    _init_megatron_tp1()
    LoRA = _import_lora()

    # ---- 检查 1+2+3：裸 target，看命名 / 冻结 / 过匹配 ----
    print("\n========== 检查 1/2/3：裸 target=['linear_qkv','linear_proj'] ==========")
    model = _build_omni_like_tree()
    if torch.cuda.is_available():
        model = model.cuda()

    bare = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=16, alpha=32)
    model = bare(model, training=True)

    adapters = _adapter_param_names(model)
    trainable = _trainable_param_names(model)
    print(f"[1] adapter 参数（前若干个）:")
    for n in adapters[:6]:
        print("    ", n)

    # 断言 1：命名包含 adapter.linear_in / linear_out
    has_in = any(n.endswith("adapter.linear_in.weight") for n in adapters)
    has_out = any(n.endswith("adapter.linear_out.weight") for n in adapters)
    assert has_in and has_out, "命名不符：没找到 adapter.linear_in/linear_out"
    print("[1] PASS：adapter 命名 = adapter.linear_in/linear_out（Block 3 可按此抽取）")

    # 断言 2：可训参数全是 adapter（基座被冻结）
    non_adapter_trainable = [n for n in trainable if not _is_adapter_name(n)]
    assert not non_adapter_trainable, f"基座没被冻干净：{non_adapter_trainable[:3]}"
    print("[2] PASS：普通 LoRA 已冻结全部基座，只有 adapter 可训（无需 VLMLoRA）")

    # 断言 3：裸 target 把 LoRA 误挂到了 audio_model（过匹配）
    audio_adapters = [n for n in adapters if "audio_model" in n]
    assert audio_adapters, "本应过匹配到 audio_model，却没有——检查结构假设"
    print(f"[3] PASS（暴露风险）：裸 target 误挂到音频塔 {len(audio_adapters)} 个 adapter:")
    for n in audio_adapters:
        print("    ", n)

    # ---- 检查 4：通配符 scoped target，音频塔不再被挂 ----
    print("\n========== 检查 4：scoped target=['*language_model*linear_qkv', ...] ==========")
    model2 = _build_omni_like_tree()
    if torch.cuda.is_available():
        model2 = model2.cuda()

    scoped = LoRA(
        target_modules=["*language_model*linear_qkv", "*language_model*linear_proj"],
        dim=16,
        alpha=32,
    )
    model2 = scoped(model2, training=True)
    adapters2 = _adapter_param_names(model2)
    audio_adapters2 = [n for n in adapters2 if "audio_model" in n]
    lang_adapters2 = [n for n in adapters2 if "language_model" in n]

    assert not audio_adapters2, f"通配符仍误挂到音频塔：{audio_adapters2}"
    assert lang_adapters2, "通配符把 language_model 也漏掉了"
    print(f"[4] PASS：scoped target 只挂到 language_model（{len(lang_adapters2)} 个），音频塔 0 个")

    print("\n================ 结论 ================")
    print("Block 4 方案确认：普通 LoRA + 通配符 target（限定 language_model）。")
    print("adapter 权重名 adapter.linear_in/out 已确认，Block 3 转换器照此实现。")
    print("=====================================")


if __name__ == "__main__":
    main()
