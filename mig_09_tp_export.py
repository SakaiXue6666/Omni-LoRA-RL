"""探针八 —— TP=2 下的 adapter 导出 parity（容器内 torchrun 起 2 进程，2 卡）。

探针七只验了 TP=1。v1 恰恰在 TP>1 的 adapter gather 上流过血（坑 16：TP=4 手写
all_gather 撞 CUDA illegal access），当年专门留了 verify_lora_tp.py（TP=2）和
verify_tp_gather.py（TP=4）。v2 把 gather 交给了 bridge，大概率没事 —— 但"大概率"
不算验过，而 4×A100 跑到一半才炸的排查成本比两张卡高得多。

判据设计上有个坑要绕开：**不能拿 TP=1 和 TP=2 的导出直接比数值**。Megatron 的
TP 初始化按 rank 分 RNG 种子，同一个逻辑权重在 TP=1 和 TP=2 下本来就不是一份随机
数，比出来的差异毫无意义。所以改成两条自洽判据：

  1. 形状必须是**完整尺寸**（q_proj.lora_B 是 [4096, 32] 而不是 [2048, 32]）。
     gather 要是没做或只拿了本 rank 的分片，这里立刻露馅。
  2. 导出的 q/k/v_proj.lora_B 三块拼起来，必须和我们**手工 all_gather** 出来的
     fused linear_qkv.adapter.linear_out 是同一批行（多重集相等）。这一条同时覆盖
     gather 有没有丢数据、以及 GQA 的 de-interleave 有没有错位 —— 后者正是 v1 要
     手写 _reorder_qkv_lora_b() 的地方。

跑法：由 modal_probe_tp_export.py 用 torchrun --nproc_per_node=2 拉起。
"""

from __future__ import annotations

import os
import traceback


OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj"]
LORA_DROPOUT = 0.0

NUM_LAYERS = 2
TP_SIZE = int(os.environ.get("PROBE_TP_SIZE", "2"))

# 探针七在 TP=1 上实测到的完整形状，作为这次的对照基线。
EXPECTED_SHAPES = {
    "q_proj.lora_A.weight": (LORA_RANK, 2048),
    "q_proj.lora_B.weight": (4096, LORA_RANK),
    "k_proj.lora_A.weight": (LORA_RANK, 2048),
    "k_proj.lora_B.weight": (512, LORA_RANK),
    "v_proj.lora_A.weight": (LORA_RANK, 2048),
    "v_proj.lora_B.weight": (512, LORA_RANK),
    "o_proj.lora_A.weight": (LORA_RANK, 4096),
    "o_proj.lora_B.weight": (2048, LORA_RANK),
}

_failures: list[str] = []
_rank = 0


def _p(msg: str) -> None:
    if _rank == 0:
        print(msg, flush=True)


def _ok(tag: str, msg: str) -> None:
    _p(f"  [PASS] {tag}: {msg}")


def _fail(tag: str, msg: str) -> None:
    _p(f"  [FAIL] {tag}: {msg}")
    _failures.append(tag)


def _banner(title: str) -> None:
    _p(f"\n===== {title} =====")


def init_distributed() -> tuple[int, int]:
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    if rank == 0:
        print(f"  world={world} tp={TP_SIZE}  dist + parallel_state 就绪", flush=True)
    return rank, world


def build_model():
    import torch

    import relax.models.qwen_omni.qwen3_omni_bridge  # noqa: F401
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(OMNI_CKPT, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    provider.num_layers = NUM_LAYERS
    freq = getattr(provider, "moe_layer_freq", None)
    if isinstance(freq, list):
        provider.moe_layer_freq = freq[:NUM_LAYERS]

    provider.tensor_model_parallel_size = TP_SIZE
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.fp16 = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()

    _p(f"  provider 就绪：层数={provider.num_layers} tp={provider.tensor_model_parallel_size} "
       f"sequence_parallel={getattr(provider, 'sequence_parallel', None)}")

    model = provider.provide(pre_process=True, post_process=True)
    return bridge, model


def apply_lora(model):
    from types import SimpleNamespace

    from relax.utils.megatron_peft_utils import build_lora_peft

    args = SimpleNamespace(
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
    )
    return build_lora_peft(args)(model, training=True)


def local_shard_report(model) -> dict:
    """本 rank 的 adapter 分片形状，以及 Megatron 打在参数上的 TP 元数据。

    linear_out（也就是 lora_B）是列并行，沿输出维切；linear_in（lora_A）通常复制。
    先把实际情况打出来，再据此判断 gather 该长什么样。
    """
    import torch.distributed as dist

    info = {}
    for name, p in model.named_parameters():
        if "adapter" not in name:
            continue
        info[name] = {
            "shape": tuple(p.shape),
            "tp": bool(getattr(p, "tensor_model_parallel", False)),
            "dim": getattr(p, "partition_dim", -1),
        }
    if dist.get_rank() == 0:
        print("  本 rank 的 adapter 分片：", flush=True)
        for n, d in list(info.items())[:4]:
            print(f"    {n}  {d['shape']}  tp={d['tp']} partition_dim={d['dim']}", flush=True)
    return info


def manual_gather_fused_qkv_b(model, layer: int = 0):
    """手工把第 0 层 linear_qkv.adapter.linear_out 沿分片维 all_gather 成完整矩阵。

    这是我们自己算的"标准答案"，用来对账 bridge 导出的 q/k/v 三块。
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state

    target = f"language_model.decoder.layers.{layer}.self_attention.linear_qkv.adapter.linear_out.weight"
    local = None
    for name, p in model.named_parameters():
        if name == target:
            local = p.detach().to(torch.float32)
            break
    if local is None:
        return None, target

    group = parallel_state.get_tensor_model_parallel_group()
    world = parallel_state.get_tensor_model_parallel_world_size()
    bufs = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(bufs, local.contiguous(), group=group)
    return torch.cat(bufs, dim=0).cpu(), target


def check_export(bridge, model, fused_full, fused_name) -> None:
    import torch

    from relax.utils import megatron_bridge_utils

    with megatron_bridge_utils.patch_megatron_model([model]):
        items = list(bridge.export_adapter_weights([model], cpu=True, show_progress=False))

    if not items:
        _fail("export", "导出为空")
        return

    exported = {it.param_name: it.weight.detach().to(torch.float32) for it in items}
    _p(f"  导出 {len(exported)} 个张量（TP={TP_SIZE}，每 rank 都跑了这次 collective）")

    # 判据 1：形状必须是完整尺寸，不能是本 rank 的分片
    bad = []
    for name, w in exported.items():
        suffix = ".".join(name.split(".")[-3:])
        want = EXPECTED_SHAPES.get(suffix)
        if want is not None and tuple(w.shape) != want:
            bad.append(f"{name}: {tuple(w.shape)} != {want}")
    if bad:
        _fail("shape", f"{len(bad)} 个张量形状不是完整尺寸，例如 {bad[:3]}")
    else:
        _ok("shape", f"全部为完整尺寸，与探针七的 TP=1 结果一致（q_proj.lora_B = {EXPECTED_SHAPES['q_proj.lora_B.weight']}）")

    if len(exported) != NUM_LAYERS * 8:
        _fail("count", f"导出 {len(exported)} 个，预期 {NUM_LAYERS * 8}")
    else:
        _ok("count", f"{len(exported)} 个 = {NUM_LAYERS} 层 × 8")

    # 判据 2：q/k/v 的 lora_B 拼起来，必须和手工 gather 的 fused 矩阵是同一批数
    if fused_full is None:
        _fail("parity", f"本 rank 找不到 {fused_name}，无法对账")
        return

    prefix = "thinker.model.layers.0.self_attn"
    try:
        q = exported[f"{prefix}.q_proj.lora_B.weight"]
        k = exported[f"{prefix}.k_proj.lora_B.weight"]
        v = exported[f"{prefix}.v_proj.lora_B.weight"]
    except KeyError as exc:
        _fail("parity", f"导出里缺 {exc}")
        return

    concat = torch.cat([q, k, v], dim=0)
    _p(f"  手工 gather 的 fused linear_out: {tuple(fused_full.shape)}")
    _p(f"  导出的 q+k+v 拼接:              {tuple(concat.shape)}")

    if concat.shape != fused_full.shape:
        _fail("parity", f"拼接形状 {tuple(concat.shape)} != fused {tuple(fused_full.shape)}")
        return

    a = torch.sort(concat.flatten()).values
    b = torch.sort(fused_full.flatten()).values
    if torch.allclose(a, b, atol=1e-3, rtol=0):
        _ok("parity", "q/k/v 拼接与手工 gather 的 fused 矩阵是同一批数 —— gather 没丢数据")
    else:
        diff = (a - b).abs().max().item()
        _fail("parity", f"多重集不等，最大差 {diff:.4f} —— gather 丢了数据或掺了别的东西")
        return

    # 逐行对账：de-interleave 是不是把行搬到了对的地方
    fused_rows = {tuple(r.tolist()) for r in fused_full}
    q_rows = {tuple(r.tolist()) for r in q}
    stray = q_rows - fused_rows
    if stray:
        _fail("deinterleave", f"q_proj 里有 {len(stray)} 行不在 fused 矩阵中")
    else:
        _ok("deinterleave", "q_proj 的每一行都能在 fused 矩阵里找到 —— GQA 拆分没错位")

    if torch.isnan(concat).any() or torch.isinf(concat).any():
        _fail("finite", "导出里有 NaN/Inf")
    else:
        _ok("finite", "导出没有 NaN/Inf")


def main() -> int:
    global _rank

    import torch

    _rank_local = int(os.environ.get("RANK", "0"))
    _rank = _rank_local

    _banner("0. 环境")
    _p(f"  torch {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}")

    _banner("1. 初始化（TP=2）")
    _rank, world = init_distributed()
    if world != TP_SIZE:
        _fail("world", f"world_size={world} 与 TP_SIZE={TP_SIZE} 不符")
        return 1

    _banner("2. 建模型 + 注入 LoRA")
    try:
        bridge, model = build_model()
        model = apply_lora(model)
        free, total = torch.cuda.mem_get_info()
        _p(f"  显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB 空闲")
        _ok("build", "TP=2 下模型与 LoRA 就绪")
    except Exception as exc:  # noqa: BLE001
        _fail("build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("3. 本 rank 的分片情况")
    local_shard_report(model)

    _banner("4. 手工 all_gather 作为标准答案")
    try:
        fused_full, fused_name = manual_gather_fused_qkv_b(model)
        if fused_full is not None:
            _ok("manual_gather", f"{fused_name} -> {tuple(fused_full.shape)}")
    except Exception as exc:  # noqa: BLE001
        _fail("manual_gather", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("5. bridge 导出并对账")
    try:
        check_export(bridge, model, fused_full, fused_name)
    except Exception as exc:  # noqa: BLE001
        _fail("export", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    _banner("结论")
    if _failures:
        _p(f"  失败项: {sorted(set(_failures))}")
        return 1
    _p("  TP=2 的 adapter 导出与 TP=1 同形状，且与手工 gather 对得上 —— v1 坑 16 那条路 v2 走得通。")
    return 0


if __name__ == "__main__":
    import sys

    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(rc)
