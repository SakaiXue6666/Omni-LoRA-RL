"""验证 M:TP>1 下 Megatron-Bridge LoRA adapter 的 TP 切分能被正确复原。

验证 K 只覆盖 TP=1;Block 3 的同步在 TP>1 时依赖「adapter 权重带正确的 TP 切分
属性(tensor_model_parallel / partition_dim),再用全量同步同款 all-gather 拼回完整
张量」。这一步是 Block 3 唯一没在低成本下验过的残留风险点,本脚本把它钉死。

做法(TP=2,torchrun --nproc_per_node=2):
  1. 用真实 Megatron attention 块(ColumnParallel linear_qkv / RowParallel linear_proj)
     建 language_model,挂 Megatron-Bridge LoRA。
  2. 逐个 adapter 权重,打印 shape / tensor_model_parallel / partition_dim。
  3. 对【被 TP 切分】的 adapter:把每个元素填成它的「全局行号」(rank*local + i),
     用与 common.all_gather_param 普通路径等价的 gather 拼回,断言完整张量沿切分维
     == 0,1,2,... —— 证明拼接顺序/位置完全正确。
  4. 对【复制(非 TP)】的 adapter:断言各 rank 的副本逐位相等 —— 证明「直接取本
     rank 副本即完整张量」成立。

两类都过 => all_gather_param 对 adapter 必然产出正确的完整权重,TP 风险清零。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


def _init_megatron_tp(tp_size: int) -> int:
    from megatron.core import parallel_state

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)
    return local_rank


def _build_attention_block(tp_size: int) -> nn.Module:
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
        num_query_groups=4,  # GQA;8 头 / 4 组,TP=2 下每 rank 2 组
        ffn_hidden_size=512,
        kv_channels=32,
        use_cpu_initialization=False,
        bf16=False,
        fp16=False,
        sequence_parallel=False,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        add_bias_linear=False,
    )
    try:
        spec = get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True)
        layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
        _print0("[build] 使用 TransformerEngine spec")
    except Exception as exc:  # noqa: BLE001
        _print0(f"[build] TE spec 失败({exc}),回退 local spec")
        spec = get_gpt_layer_local_spec(qk_layernorm=True)
        layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
    layer.config = config
    return layer


def _build_language_model(tp_size: int) -> nn.Module:
    root = nn.Module()
    root.thinker = nn.Module()
    root.thinker.language_model = nn.Module()
    root.thinker.language_model.decoder = nn.Module()
    root.thinker.language_model.decoder.layers = nn.ModuleList([_build_attention_block(tp_size)])
    return root


def _import_lora():
    try:
        from megatron.bridge.peft.lora import LoRA

        return LoRA
    except Exception:  # noqa: BLE001
        from megatron.bridge.peft import LoRA  # type: ignore

        return LoRA


def _print0(*a) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


def _gather_like_production(param: torch.Tensor) -> torch.Tensor:
    """等价于 relax common.all_gather_param 的普通路径(那边 line 99-113):
    沿 partition_dim all_gather 后 cat。adapter 不含 linear_fc1/fc2/conv1d/experts,
    不会进任何特殊分支,故这段就是生产代码对 adapter 的实际行为。
    """
    from megatron.core import parallel_state as mpu

    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    parts = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(parts, param.data, group=tp_group)
    assert getattr(param, "partition_stride", 1) == 1, "partition_stride != 1 不支持"
    return torch.cat(parts, dim=param.partition_dim)


def _is_adapter_name(n: str) -> bool:
    return ".adapter." in n


def main() -> None:
    tp_size = int(os.environ.get("TP_SIZE", "2"))
    _init_megatron_tp(tp_size)
    from megatron.core import parallel_state as mpu

    rank = dist.get_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    LoRA = _import_lora()

    model = _build_language_model(tp_size).cuda()
    lora = LoRA(target_modules=["linear_qkv", "linear_proj"], dim=16, alpha=32)
    model = lora(model, training=True)

    adapters = [(n, p) for n, p in model.named_parameters() if _is_adapter_name(n)]

    _print0("\n========== adapter 权重的 TP 属性 ==========")
    for n, p in adapters:
        tp = bool(getattr(p, "tensor_model_parallel", False))
        pd = getattr(p, "partition_dim", -1)
        _print0(f"  {n}\n      shape(local)={tuple(p.shape)} tensor_model_parallel={tp} partition_dim={pd}")

    failures: list[str] = []
    n_sharded = 0
    n_replicated = 0

    for n, p in adapters:
        tp = bool(getattr(p, "tensor_model_parallel", False))
        pd = getattr(p, "partition_dim", -1)

        if tp and pd is not None and pd >= 0:
            # ---- 切分:填全局行号 -> gather -> 断言 0,1,2,... ----
            n_sharded += 1
            local_size = p.shape[pd]
            idx = torch.arange(local_size, device=p.device, dtype=p.dtype)
            view_shape = [1] * p.dim()
            view_shape[pd] = local_size
            fill = (tp_rank * local_size + idx.view(view_shape)).expand_as(p)
            with torch.no_grad():
                p.data.copy_(fill)

            full = _gather_like_production(p)
            gsize = full.shape[pd]
            exp_shape = [1] * full.dim()
            exp_shape[pd] = gsize
            expected = torch.arange(gsize, device=full.device, dtype=full.dtype).view(exp_shape).expand_as(full)
            ok = torch.equal(full, expected)
            _print0(
                f"  [shard] {n.split('.')[-3]}.{n.split('.')[-2]} "
                f"local{tuple(p.shape)} -> full{tuple(full.shape)} 沿 dim{pd} 顺序: {'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{n} gather 顺序错")
        else:
            # ---- 复制:断言各 rank 逐位相等 ----
            n_replicated += 1
            parts = [torch.empty_like(p.data) for _ in range(tp_size)]
            dist.all_gather(parts, p.data.contiguous(), group=mpu.get_tensor_model_parallel_group())
            same = all(torch.equal(parts[0], q) for q in parts[1:])
            _print0(
                f"  [replic] {n.split('.')[-3]}.{n.split('.')[-2]} "
                f"shape{tuple(p.shape)} 各 rank 相等: {'OK' if same else 'FAIL'}"
            )
            if not same:
                failures.append(f"{n} 复制副本各 rank 不一致")

    # 汇总(跨 rank 收集 failure)
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    all_fail = [f for sub in gathered for f in (sub or [])]

    if rank == 0:
        print("\n================ 结论 ================", flush=True)
        print(f"adapter 总数={len(adapters)}  TP切分={n_sharded}  复制={n_replicated}", flush=True)
        if all_fail:
            print(f"FAIL: {len(all_fail)} 项", flush=True)
            for f in sorted(set(all_fail)):
                print(f"  - {f}", flush=True)
        else:
            print("PASS:TP>1 下 adapter 切分能被 all_gather 正确复原,Block 3 的 TP 风险清零。", flush=True)
        print("=====================================", flush=True)

    dist.barrier()
    if all_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
