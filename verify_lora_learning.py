"""机制层验证：LoRA 真的在学吗？

这个脚本【不加载真实 30B 权重、不起 sglang/Ray、不跑完整 RL】，
只用真实的 Megatron 模块搭一个最小 Qwen3-Omni-like 结构，挂上 scoped LoRA，
跑几步真实的 forward + backward + optimizer.step()，然后逐条核对：

  机制层断言（这是 PEFT-LoRA RL 能学的最底层前提）：
    A. adapter（lora_A=linear_in / lora_B=linear_out）确实拿到梯度
    B. optimizer.step() 后 adapter 权重 delta 非零        -> “在学”
    C. 基座(base)参数 requires_grad=False、grad 始终为 None
    D. optimizer.step() 后 base 权重 delta == 0           -> base 冻结没被误动

LoRA 初始化特性（务必知道，否则会误判“没在学”）：
    linear_out(=lora_B) 零初始化 -> 第 1 步就有梯度、就更新；
    linear_in (=lora_A) 随机初始化，但因初始 B=0，第 1 步梯度=0，
    从第 2 步起才更新。所以这里跑 3 步，确保两者都被验证到。

运行环境：Relax 的 Megatron 训练镜像。单卡 TP=1 即可，CPU 也能跑。
    python verify_lora_learning.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


# --- 下面 4 个 builder 与 verify_lora_attach.py 保持一致，方便对照 ---

def _init_megatron_tp1() -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29502")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)


def _build_attention_block() -> nn.Module:
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
        num_query_groups=4,
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
    root = nn.Module()
    root.thinker = nn.Module()
    root.thinker.language_model = nn.Module()
    root.thinker.language_model.decoder = nn.Module()
    root.thinker.language_model.decoder.layers = nn.ModuleList([_build_attention_block()])
    root.thinker.audio_model = _build_attention_block()
    return root


def _is_adapter_name(n: str) -> bool:
    return ".adapter." in n or ".lora_a" in n or ".lora_b" in n


def _import_lora():
    try:
        from megatron.bridge.peft.lora import LoRA

        return LoRA
    except Exception:  # noqa: BLE001
        from megatron.bridge.peft import LoRA  # type: ignore

        return LoRA


def _forward_loss(model: nn.Module, hidden_size: int, device, dtype) -> torch.Tensor:
    """跑一次真实 forward，返回一个依赖 adapter 输出的标量 loss。

    优先跑整层 TransformerLayer.forward（最接近真实路径，自动把 qkv/proj 的维度对上）；
    若该版本的 forward 需要额外位置编码等参数而失败，则回退到直接调用被 LoRA 包过的
    linear_qkv（仍然能让 adapter 进入 autograd 图）。
    """
    layer = model.thinker.language_model.decoder.layers[0]
    seq, batch = 8, 2
    # Megatron 约定 sequence-first: [seq, batch, hidden]
    hidden = torch.randn(seq, batch, hidden_size, device=device, dtype=dtype)

    try:
        out = layer(hidden, attention_mask=None)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.float().pow(2).mean()
    except Exception as exc:  # noqa: BLE001
        print(f"[forward] 整层 forward 失败({type(exc).__name__}: {exc})，回退到直接调 linear_qkv")
        qkv = layer.self_attention.linear_qkv
        x = torch.randn(seq, batch, hidden_size, device=device, dtype=dtype)
        out = qkv(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.float().pow(2).mean()


def main() -> None:
    _init_megatron_tp1()
    LoRA = _import_lora()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    hidden_size = 256
    n_steps = 3

    print("\n========== 搭最小 Omni 树 + 挂 scoped LoRA(language_model) ==========")
    model = _build_omni_like_tree().to(device)
    scoped = LoRA(
        target_modules=["*language_model*linear_qkv", "*language_model*linear_proj"],
        dim=16,
        alpha=32,
    )
    model = scoped(model, training=True)
    model = model.to(device)

    adapter_params = {n: p for n, p in model.named_parameters() if _is_adapter_name(n)}
    base_params = {n: p for n, p in model.named_parameters() if not _is_adapter_name(n)}
    assert adapter_params, "没挂上任何 adapter，结构/版本有问题"
    print(f"  adapter 参数 {len(adapter_params)} 个，base 参数 {len(base_params)} 个")

    # 断言 C(静态部分)：base 全部 requires_grad=False，adapter 全部可训
    base_trainable = [n for n, p in base_params.items() if p.requires_grad]
    adapter_frozen = [n for n, p in adapter_params.items() if not p.requires_grad]
    assert not base_trainable, f"base 没冻干净: {base_trainable[:3]}"
    assert not adapter_frozen, f"有 adapter 被冻了: {adapter_frozen[:3]}"
    print("  [C-静态] PASS：base 全冻结(requires_grad=False)，adapter 全可训")

    # 快照初始权重
    adapter_init = {n: p.detach().clone() for n, p in adapter_params.items()}
    base_init = {n: p.detach().clone() for n, p in base_params.items()}

    # 只把可训(adapter)参数交给 optimizer——和真实 PEFT 训练一致
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-2)

    print(f"\n========== 跑 {n_steps} 步 forward+backward+step ==========")
    base_grad_seen = False  # 是否出现过 base.grad 非 None（不该出现）
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss = _forward_loss(model, hidden_size, device, dtype)
        loss.backward()

        # 断言 A：至少一个 adapter 拿到非零梯度
        ada_grad_norm = sum(
            p.grad.norm().item() for p in adapter_params.values() if p.grad is not None
        )
        # 断言 C(动态部分)：base 参数 grad 必须始终为 None
        step_base_grad = [n for n, p in base_params.items() if p.grad is not None]
        if step_base_grad:
            base_grad_seen = True
            print(f"  step {step}: ⚠️ base 出现梯度: {step_base_grad[:3]}")

        opt.step()
        print(f"  step {step}: loss={loss.item():.6f}  adapter_grad_norm={ada_grad_norm:.6e}")

    # 计算 delta
    def _delta(cur: dict, init: dict) -> dict:
        return {n: (cur[n].detach() - init[n]).norm().item() for n in init}

    adapter_delta = _delta(adapter_params, adapter_init)
    base_delta = _delta(base_params, base_init)

    # 按 lora_A(linear_in) / lora_B(linear_out) 分组看 delta
    a_changed = [n for n, d in adapter_delta.items() if "linear_in" in n and d > 0]
    b_changed = [n for n, d in adapter_delta.items() if "linear_out" in n and d > 0]
    base_changed = [n for n, d in base_delta.items() if d > 0]

    print("\n========== 结果 ==========")
    print(f"  adapter 中 linear_in (lora_A) 权重发生变化的: {len(a_changed)} 个")
    print(f"  adapter 中 linear_out(lora_B) 权重发生变化的: {len(b_changed)} 个")
    print(f"  base 权重发生变化的: {len(base_changed)} 个 (应为 0)")
    max_ada = max(adapter_delta.values()) if adapter_delta else 0.0
    max_base = max(base_delta.values()) if base_delta else 0.0
    print(f"  adapter delta 最大值={max_ada:.6e}  base delta 最大值={max_base:.6e}")

    # ---- 最终断言 ----
    assert b_changed, "断言B失败：lora_B(linear_out) 一步都没更新 -> LoRA 没在学"
    assert a_changed, "断言B失败：lora_A(linear_in) 没更新（跑满 3 步后仍为 0？）"
    assert not base_changed, f"断言D失败：base 权重被改动了 {base_changed[:3]}"
    assert not base_grad_seen, "断言C失败：base 参数出现过梯度（应被冻结）"

    print("\n================ 结论 ================")
    print("[PASS] 机制层验证通过：")
    print("  A. adapter 拿到非零梯度")
    print("  B. lora_A/lora_B 权重经 optimizer.step() 后确实更新 -> LoRA 在学")
    print("  C. base 参数 grad 始终为 None")
    print("  D. base 权重 delta == 0 -> 基座冻结没被误动")
    print("=====================================")


if __name__ == "__main__":
    main()
