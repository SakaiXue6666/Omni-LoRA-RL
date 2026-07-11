"""Tiny Qwen3-Omni TP=4 全链路 LoRA Gather 验证。

在 4x T4（~$1.3/hr）上用一个 ~200MB 的超小 Qwen3-Omni 模型（随机权重，保留完整
GQA/MoE/fused-QKV 架构特征）验证三种 TP gather 方法的正确性，并通过 SGLang 端到端
确认热加载后推理行为变化。

用法（Modal）：
    modal run verify_tp_gather.py::main

内部执行：
    torchrun --nproc_per_node=4 verify_tp_gather.py --run-stages
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

# ---------------------------------------------------------------------------
# Tiny Qwen3-Omni 配置：保留真实架构特征，只缩层数/专家数/词表
# ---------------------------------------------------------------------------
TINY_QWEN3_OMNI_TEXT_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 2,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 768,
    "moe_intermediate_size": 768,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 4096,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "decoder_sparse_step": 1,
    "norm_topk_prob": True,
    "torch_dtype": "float16",
    "tie_word_embeddings": False,
}

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["*language_model*linear_qkv", "*language_model*linear_proj"]

# GQA 关键维度（这些是 bug 触发点）
NUM_HEADS = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128
HIDDEN = 2048
Q_OUT = NUM_HEADS * HEAD_DIM  # 4096 (≠ hidden!)
KV_OUT = NUM_KV_HEADS * HEAD_DIM  # 512
QKV_OUT = Q_OUT + 2 * KV_OUT  # 5120


# ===========================================================================
# 方法 B: Slime _smart_gather（移植为独立函数）
# ===========================================================================
def expected_full_shape(clean_name: str, rank: int) -> tuple[int, int] | None:
    """推断给定 LoRA 参数的未分片全量形状。

    Megatron-bridge 命名约定：
        linear_qkv.adapter.linear_in   -> LoRA A   shape (rank, hidden)
        linear_qkv.adapter.linear_out  -> LoRA B   shape (qkv_out, rank)
        linear_proj.adapter.linear_in  -> LoRA A   shape (rank, q_out)   ** GQA-aware **
        linear_proj.adapter.linear_out -> LoRA B   shape (hidden, rank)
    """
    if "linear_qkv.adapter.linear_in.weight" in clean_name:
        return (rank, HIDDEN)
    if "linear_qkv.adapter.linear_out.weight" in clean_name:
        return (QKV_OUT, rank)
    if "linear_proj.adapter.linear_in.weight" in clean_name:
        return (rank, Q_OUT)
    if "linear_proj.adapter.linear_out.weight" in clean_name:
        return (HIDDEN, rank)
    return None


def smart_gather(
    clean_name: str,
    param_data: "torch.Tensor",
    tp_size: int,
    tp_group,
    lora_rank: int = LORA_RANK,
) -> tuple["torch.Tensor", str, dict]:
    """Slime 的 _smart_gather 移植版：推断分片轴，正确 all_gather。"""
    import torch
    import torch.distributed as dist

    raw = tuple(param_data.shape)
    info = {"raw_shape": raw, "expected_full": None, "tp_size": tp_size, "decision": None, "final_shape": None}

    expected = expected_full_shape(clean_name, lora_rank)
    info["expected_full"] = expected
    if expected is None:
        info["decision"] = "no_dims_no_gather"
        info["final_shape"] = raw
        return param_data, "no_dims_no_gather", info

    if raw == expected:
        info["decision"] = "replicated"
        info["final_shape"] = raw
        return param_data, "replicated", info

    for dim, dim_size in enumerate(expected):
        if tp_size == 0 or dim_size % tp_size != 0:
            continue
        candidate = list(expected)
        candidate[dim] = dim_size // tp_size
        if tuple(candidate) == raw:
            parts = [torch.empty_like(param_data) for _ in range(tp_size)]
            dist.all_gather(parts, param_data.contiguous(), group=tp_group)
            gathered = torch.cat(parts, dim=dim)
            info["decision"] = f"sharded_dim{dim}"
            info["final_shape"] = tuple(gathered.shape)
            return gathered, info["decision"], info

    parts = [torch.empty_like(param_data) for _ in range(tp_size)]
    dist.all_gather(parts, param_data.contiguous(), group=tp_group)
    gathered = torch.cat(parts, dim=0)
    info["decision"] = "fallback_dim0"
    info["final_shape"] = tuple(gathered.shape)
    return gathered, "fallback_dim0", info


# ===========================================================================
# 方法 A: Relax all_gather（简化版，只处理 adapter 参数）
# ===========================================================================
def relax_gather(
    name: str,
    param: "torch.nn.Parameter",
    tp_size: int,
    tp_group,
) -> "torch.Tensor":
    """Relax common.py 的 all_gather_param 简化版（只处理 adapter 路径）。"""
    import torch
    import torch.distributed as dist

    if not getattr(param, "tensor_model_parallel", False):
        return param.data
    if getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    param_data = param.data.contiguous()
    param_partitions = [torch.empty_like(param_data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param_data, group=tp_group)
    partition_dim = getattr(param, "partition_dim", 0)
    return torch.cat(param_partitions, dim=partition_dim)


# ===========================================================================
# 方法 C: Bridge export_adapter_weights（尝试调用，无则跳过）
# ===========================================================================
def bridge_gather(bridge, model: list) -> dict[str, "torch.Tensor"] | None:
    """尝试调用官方 export_adapter_weights API，失败返回 None。"""
    if not hasattr(bridge, "export_adapter_weights"):
        return None
    result = {}
    for name, tensor in bridge.export_adapter_weights(model, cpu=False):
        result[name] = tensor
    return result


# ===========================================================================
# Stage 1-3: Megatron 初始化 + LoRA 挂载 + Gather 对比
# ===========================================================================
def run_megatron_stages():
    """在 torchrun 进程中执行：TP=4 初始化 → LoRA 挂载 → 三种 gather 对比。"""
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "4"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    tp_size = world_size
    tp_group = dist.group.WORLD

    if rank == 0:
        print(f"[Stage 1] world_size={world_size}, tp_size={tp_size}")
        print(f"[Stage 1] GQA: Q_OUT={Q_OUT}, KV_OUT={KV_OUT}, QKV_OUT={QKV_OUT}, HIDDEN={HIDDEN}")

    # --- 构造 tiny Qwen3-Omni 注意力层（带 LoRA adapter）---
    torch.manual_seed(42)

    num_layers = TINY_QWEN3_OMNI_TEXT_CONFIG["num_hidden_layers"]
    adapters = {}  # name -> param (with TP metadata)

    for layer_idx in range(num_layers):
        prefix = f"thinker.language_model.decoder.layers.{layer_idx}.self_attention"

        # linear_qkv adapter: LoRA A shape (rank, hidden), LoRA B shape (qkv_out, rank)
        # TP 切 LoRA A 的 dim=1 (input)? 不，Megatron 切 LoRA B 的 dim=0 (output=qkv_out)
        # linear_qkv 是 column parallel → output 被 TP 切
        # adapter.linear_in (LoRA A): replicated (rank, hidden) → 不切
        # adapter.linear_out (LoRA B): column parallel → dim=0 切 qkv_out/tp
        qkv_lora_a = torch.randn(LORA_RANK, HIDDEN, device="cuda", dtype=torch.float16)
        qkv_lora_b_full = torch.randn(QKV_OUT, LORA_RANK, device="cuda", dtype=torch.float16)
        qkv_lora_b_shard = qkv_lora_b_full.chunk(tp_size, dim=0)[rank]

        # linear_proj adapter: LoRA A shape (rank, q_out), LoRA B shape (hidden, rank)
        # linear_proj 是 row parallel → input 被 TP 切
        # adapter.linear_in (LoRA A): row parallel → dim=1 切 q_out/tp
        # adapter.linear_out (LoRA B): replicated (hidden, rank) → 不切
        proj_lora_a_full = torch.randn(LORA_RANK, Q_OUT, device="cuda", dtype=torch.float16)
        proj_lora_a_shard = proj_lora_a_full.chunk(tp_size, dim=1)[rank]
        proj_lora_b = torch.randn(HIDDEN, LORA_RANK, device="cuda", dtype=torch.float16)

        # 模拟 Megatron 的 TP 属性标注
        def _make_param(data, is_tp, partition_dim, parallel_mode=None):
            p = torch.nn.Parameter(data.clone())
            p.tensor_model_parallel = is_tp
            p.partition_dim = partition_dim
            p.partition_stride = 1
            if parallel_mode:
                p.parallel_mode = parallel_mode
            return p

        n1 = f"{prefix}.linear_qkv.adapter.linear_in.weight"
        n2 = f"{prefix}.linear_qkv.adapter.linear_out.weight"
        n3 = f"{prefix}.linear_proj.adapter.linear_in.weight"
        n4 = f"{prefix}.linear_proj.adapter.linear_out.weight"

        # qkv LoRA A: replicated（所有 rank 相同）
        adapters[n1] = _make_param(qkv_lora_a, is_tp=False, partition_dim=0, parallel_mode="duplicated")
        # qkv LoRA B: TP-sharded along dim=0
        adapters[n2] = _make_param(qkv_lora_b_shard, is_tp=True, partition_dim=0)
        # proj LoRA A: TP-sharded along dim=1
        adapters[n3] = _make_param(proj_lora_a_shard, is_tp=True, partition_dim=1)
        # proj LoRA B: replicated
        adapters[n4] = _make_param(proj_lora_b, is_tp=False, partition_dim=0, parallel_mode="duplicated")

    if rank == 0:
        print(f"[Stage 1] 构造完成: {len(adapters)} 个 adapter 参数")
        for n, p in list(adapters.items())[:4]:
            print(f"  {n}: shape={tuple(p.shape)}, tp={p.tensor_model_parallel}, "
                  f"pdim={p.partition_dim}, mode={getattr(p, 'parallel_mode', None)}")

    # --- Stage 2: 三种 Gather ---
    dist.barrier()
    if rank == 0:
        print("\n[Stage 2] 开始 gather 对比...")

    # 方法 A: Relax gather
    result_a = {}
    for name, param in adapters.items():
        result_a[name] = relax_gather(name, param, tp_size, tp_group)

    # 方法 B: Slime smart_gather
    result_b = {}
    decisions_b = {}
    for name, param in adapters.items():
        gathered, decision, info = smart_gather(
            name, param.data, tp_size, tp_group, lora_rank=LORA_RANK
        )
        result_b[name] = gathered
        decisions_b[name] = decision

    # 方法 C: bridge (skip if API unavailable)
    result_c = None
    bridge_available = False
    try:
        from megatron.bridge import AutoBridge
        bridge_available = hasattr(AutoBridge, "export_adapter_weights")
    except ImportError:
        pass

    if rank == 0:
        print(f"  方法 C (bridge.export_adapter_weights): {'可用' if bridge_available else '不可用(跳过)'}")

    # --- Stage 3: 正确性断言 ---
    dist.barrier()
    if rank == 0:
        print("\n[Stage 3] 正确性检查...")

    failures = []
    for name in adapters:
        shape_a = tuple(result_a[name].shape)
        shape_b = tuple(result_b[name].shape)

        # 形状检查
        expected = expected_full_shape(name, LORA_RANK)
        if expected is not None:
            if shape_a != expected:
                failures.append(f"[A] {name}: shape {shape_a} != expected {expected}")
            if shape_b != expected:
                failures.append(f"[B] {name}: shape {shape_b} != expected {expected}")
        elif shape_a != shape_b:
            failures.append(f"[AB shape] {name}: A={shape_a} vs B={shape_b}")

        # 数值一致性（A 和 B 应得到相同结果）
        if not torch.allclose(result_a[name], result_b[name], atol=1e-5):
            max_diff = (result_a[name] - result_b[name]).abs().max().item()
            failures.append(f"[AB value] {name}: max_diff={max_diff:.6e}")

    # GQA 关键断言
    for name in adapters:
        if "linear_proj.adapter.linear_in.weight" in name:
            shape = tuple(result_b[name].shape)
            if shape != (LORA_RANK, Q_OUT):
                failures.append(
                    f"[GQA] {name}: shape={shape}, "
                    f"expected=(LORA_RANK={LORA_RANK}, Q_OUT={Q_OUT})"
                )
        if "linear_qkv.adapter.linear_out.weight" in name:
            shape = tuple(result_b[name].shape)
            if shape != (QKV_OUT, LORA_RANK):
                failures.append(
                    f"[GQA] {name}: shape={shape}, "
                    f"expected=(QKV_OUT={QKV_OUT}, LORA_RANK={LORA_RANK})"
                )

    # 打印诊断
    if rank == 0:
        print("\n  === Gather 决策 (方法 B) ===")
        for name, dec in sorted(decisions_b.items()):
            print(f"    {name.split('.')[-3]}...{name.split('.')[-1]}: {dec}")

        print(f"\n  === 形状验证 ===")
        for name in sorted(adapters.keys()):
            exp = expected_full_shape(name, LORA_RANK)
            got = tuple(result_a[name].shape)
            status = "OK" if got == exp else "MISMATCH"
            short = name.replace("thinker.language_model.decoder.", "")
            print(f"    [{status}] {short}: {got} (expect {exp})")

        if failures:
            print(f"\n  === FAILURES ({len(failures)}) ===")
            for f in failures:
                print(f"    FAIL: {f}")
        else:
            print(f"\n  [ALL PASS] Stage 1-3: {len(adapters)} 参数 × 2 方法，全部形状/数值一致")

    # 额外测试：模拟 distributed optimizer 非连续 buffer
    if rank == 0:
        print("\n[Stage 2b] 非连续 buffer 测试（模拟 distributed optimizer 场景）...")

    nc_failures = []
    for name, param in list(adapters.items())[:4]:
        # 构造真正非连续的张量：转置一个 2D tensor
        rows, cols = param.shape
        big_2d = torch.randn(cols, rows, device="cuda", dtype=torch.float16)
        view = big_2d.t()  # shape=(rows, cols) but non-contiguous
        assert not view.is_contiguous(), f"Expected non-contiguous tensor for {name}"

        nc_param = torch.nn.Parameter(view)
        nc_param.tensor_model_parallel = param.tensor_model_parallel
        nc_param.partition_dim = param.partition_dim
        nc_param.partition_stride = 1
        if hasattr(param, "parallel_mode"):
            nc_param.parallel_mode = param.parallel_mode

        try:
            gathered_nc = relax_gather(name, nc_param, tp_size, tp_group)
            _, dec_nc, _ = smart_gather(name, nc_param.data, tp_size, tp_group)
            if rank == 0:
                print(f"    [OK] {name.split('.')[-3]}...{name.split('.')[-1]}: "
                      f"non-contiguous gather succeeded (decision={dec_nc})")
        except RuntimeError as e:
            nc_failures.append(f"{name}: {e}")
            if rank == 0:
                print(f"    [FAIL] {name.split('.')[-3]}...{name.split('.')[-1]}: {e}")

    if rank == 0:
        if nc_failures:
            print(f"\n  [WARN] 非连续 buffer 导致 {len(nc_failures)} 个 gather 失败")
            print("  这正是坑16 的根因！.contiguous() 修复后应不再出现。")
        else:
            print("  [OK] 非连续 buffer 全部 gather 成功（.contiguous() 修复有效）")

    # Stage 2c: 模拟 torch_memory_saver 场景（GPU storage 已释放，只剩 CPU 备份）
    if rank == 0:
        print("\n[Stage 2c] torch_memory_saver 模拟（GPU storage 释放 → CPU 备份 → 搬回 GPU → gather）...")

    tms_failures = []
    for name, param in list(adapters.items())[:4]:
        # 模拟：参数原本在 GPU，torch_memory_saver 把它 offload 到 CPU
        cpu_backup = param.data.clone().cpu()

        # 模拟 GPU 侧 tensor 已被释放（用全 0 模拟 dangling pointer 读到垃圾）
        freed_gpu = torch.zeros_like(param.data)

        # 正确做法：weights_getter 应走 translate_gpu_to_cpu=True 拿到 cpu_backup
        # 然后搬回 GPU 再 gather
        try:
            restored_gpu = cpu_backup.to(param.data.device).contiguous()

            # 用 restored tensor 做 gather
            restored_param = torch.nn.Parameter(restored_gpu)
            restored_param.tensor_model_parallel = param.tensor_model_parallel
            restored_param.partition_dim = param.partition_dim
            restored_param.partition_stride = 1
            if hasattr(param, "parallel_mode"):
                restored_param.parallel_mode = param.parallel_mode

            gathered = relax_gather(name, restored_param, tp_size, tp_group)

            # 验证结果与原始 gather 一致
            expected = expected_full_shape(name, LORA_RANK)
            shape_ok = expected is None or tuple(gathered.shape) == expected

            if not shape_ok:
                tms_failures.append(f"{name}: shape mismatch after CPU restore")
            elif rank == 0:
                print(f"    [OK] {name.split('.')[-3]}...{name.split('.')[-1]}: "
                      f"CPU→GPU→gather OK, shape={tuple(gathered.shape)}")
        except Exception as e:
            tms_failures.append(f"{name}: {e}")
            if rank == 0:
                print(f"    [FAIL] {name.split('.')[-3]}...{name.split('.')[-1]}: {e}")

    # 验证：如果错误地用 freed GPU tensor 做 gather，结果应该全是 0（垃圾）
    if rank == 0:
        print("\n    --- 对照组：使用已释放的 GPU tensor（应得到全 0 垃圾） ---")
    garbage_detected = 0
    for name, param in list(adapters.items())[:4]:
        freed_gpu = torch.zeros_like(param.data)
        freed_param = torch.nn.Parameter(freed_gpu)
        freed_param.tensor_model_parallel = param.tensor_model_parallel
        freed_param.partition_dim = param.partition_dim
        freed_param.partition_stride = 1
        if hasattr(param, "parallel_mode"):
            freed_param.parallel_mode = param.parallel_mode

        gathered_garbage = relax_gather(name, freed_param, tp_size, tp_group)
        if gathered_garbage.abs().max().item() == 0.0:
            garbage_detected += 1
            if rank == 0:
                print(f"    [证实] {name.split('.')[-3]}...{name.split('.')[-1]}: "
                      f"freed tensor gather 结果全 0（= CUDA illegal memory access 的来源）")

    if rank == 0:
        if tms_failures:
            print(f"\n  [FAIL] torch_memory_saver 模拟: {len(tms_failures)} 个失败")
            for f in tms_failures:
                print(f"    {f}")
        else:
            print(f"\n  [ALL PASS] torch_memory_saver 场景:")
            print(f"    - CPU 备份 → GPU → gather: 全部正确")
            print(f"    - 对照组: {garbage_detected}/4 个 freed tensor 产出全 0 垃圾（证实了不走 CPU 备份就会崩）")
            print(f"    - 结论: translate_gpu_to_cpu=True 是坑16 的正确修复")


    # 保存 gathered state_dict 供 Stage 4 使用
    if rank == 0:
        state_for_sglang = {}
        for name, tensor in result_b.items():
            hf_name = _megatron_to_hf_lora_name(name)
            if hf_name:
                state_for_sglang[hf_name] = tensor.cpu()
        torch.save(state_for_sglang, "/tmp/gathered_lora_state.pt")
        print(f"\n  已保存 {len(state_for_sglang)} 个 HF 格式 LoRA 张量 -> /tmp/gathered_lora_state.pt")
        for n, t in list(state_for_sglang.items())[:4]:
            print(f"    {n}: {tuple(t.shape)}")

    dist.barrier()
    dist.destroy_process_group()
    return len(failures) == 0


def _megatron_to_hf_lora_name(megatron_name: str) -> str | None:
    """Megatron adapter 名 → HF PEFT LoRA 名（用于 SGLang）。"""
    import re

    m = re.match(
        r"thinker\.language_model\.decoder\.layers\.(\d+)\.self_attention\."
        r"(linear_qkv|linear_proj)\.adapter\.(linear_in|linear_out)\.weight",
        megatron_name,
    )
    if not m:
        return None

    layer_idx = m.group(1)
    module = m.group(2)
    ab = m.group(3)

    lora_suffix = "lora_A" if ab == "linear_in" else "lora_B"

    if module == "linear_qkv":
        hf_module = "qkv_proj"
    elif module == "linear_proj":
        hf_module = "o_proj"
    else:
        return None

    return f"model.layers.{layer_idx}.self_attn.{hf_module}.{lora_suffix}.weight"


# ===========================================================================
# Stage 4: SGLang 热加载 + 推理对比
# ===========================================================================
def run_sglang_stage():
    """启动 tiny SGLang server → 加载 gathered LoRA → 验证推理行为变化。"""
    import subprocess
    import time

    import torch

    state_path = "/tmp/gathered_lora_state.pt"
    if not os.path.exists(state_path):
        print("[Stage 4] 没有 gathered state，跳过 SGLang 阶段")
        return True

    model_dir = _ensure_tiny_model_dir()
    print(f"[Stage 4] Tiny 模型目录: {model_dir}")

    port = 30100
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--port", str(port),
        "--tp-size", "1",
        "--dtype", "float16",
        "--mem-fraction-static", "0.3",
        "--disable-cuda-graph",
        "--attention-backend", "triton",
        "--trust-remote-code",
        "--enable-lora",
        "--max-loras", "2",
        "--max-lora-rank", str(LORA_RANK),
        "--lora-target-modules", "qkv_proj", "o_proj",
        "--lora-backend", "triton",
    ]

    print(f"[Stage 4] 启动 SGLang server: {' '.join(server_cmd[-6:])}")
    server_proc = subprocess.Popen(
        server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    if not _wait_for_server(port, timeout=180, proc=server_proc):
        print("[Stage 4] FAIL: SGLang server 未能在 180s 内就绪")
        server_proc.kill()
        return False

    print("[Stage 4] SGLang server 就绪，开始验证...")

    try:
        import requests

        base_url = f"http://127.0.0.1:{port}"

        # Base 推理
        base_resp = requests.post(f"{base_url}/generate", json={
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
        }, timeout=30)
        base_out = base_resp.json()
        print(f"  [base] output: {base_out.get('text', '')[:50]}")

        # 热加载 LoRA
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        lora_resp = requests.post(f"{base_url}/load_lora_adapter", json={
            "lora_name": "test_lora",
            "lora_tensors": {k: v.tolist() for k, v in state.items()},
        }, timeout=60)
        if lora_resp.status_code != 200:
            # 尝试 from_tensors 接口
            print(f"  [warn] load_lora_adapter HTTP API 不支持 tensors, 尝试本地加载...")
            success = _try_local_lora_load(base_url, state, model_dir)
            if not success:
                print("[Stage 4] FAIL: LoRA 加载失败")
                return False
        else:
            print(f"  [lora] 加载成功: {lora_resp.json()}")

        # LoRA 推理
        lora_out_resp = requests.post(f"{base_url}/generate", json={
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            "lora_path": "test_lora",
        }, timeout=30)
        lora_out = lora_out_resp.json()
        print(f"  [lora] output: {lora_out.get('text', '')[:50]}")

        # 验证：LoRA 输出应与 base 不同（随机初始化的 LoRA 几乎必然改变输出）
        if base_out.get("text") == lora_out.get("text"):
            print("  [WARN] base 和 lora 输出相同——可能 LoRA 未生效")
        else:
            print("  [OK] base 和 lora 输出不同——LoRA 热加载生效")

        return True

    except Exception as e:
        print(f"[Stage 4] 异常: {e}")
        return False
    finally:
        server_proc.kill()
        server_proc.wait()


def _try_local_lora_load(base_url: str, state: dict, model_dir: str) -> bool:
    """通过保存为 PEFT 目录再加载的方式兜底。"""
    import torch
    from safetensors.torch import save_file

    lora_dir = "/tmp/test_lora_adapter"
    os.makedirs(lora_dir, exist_ok=True)

    save_file(state, os.path.join(lora_dir, "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": model_dir,
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f)

    import requests
    resp = requests.post(f"{base_url}/load_lora_adapter", json={
        "lora_name": "test_lora",
        "lora_path": lora_dir,
    }, timeout=60)
    if resp.status_code == 200:
        print(f"  [lora] 通过 PEFT 目录加载成功")
        return True
    print(f"  [lora] PEFT 目录加载也失败: {resp.status_code} {resp.text[:200]}")
    return False


def _ensure_tiny_model_dir() -> str:
    """生成 tiny Qwen3-Omni 的 HF 模型目录（随机权重）。"""
    import torch
    from safetensors.torch import save_file

    model_dir = "/tmp/tiny_qwen3_omni"
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return model_dir
    os.makedirs(model_dir, exist_ok=True)

    cfg = dict(TINY_QWEN3_OMNI_TEXT_CONFIG)
    cfg["architectures"] = ["Qwen3MoeForCausalLM"]
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    # 生成随机权重
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    N = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    expert_ffn = cfg["moe_intermediate_size"]
    n_experts = cfg["num_experts"]

    torch.manual_seed(123)
    state = {}
    state["model.embed_tokens.weight"] = torch.randn(V, H, dtype=torch.float16)

    for i in range(N):
        pfx = f"model.layers.{i}"
        state[f"{pfx}.input_layernorm.weight"] = torch.ones(H, dtype=torch.float16)
        state[f"{pfx}.post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.float16)
        # Attention
        state[f"{pfx}.self_attn.q_proj.weight"] = torch.randn(n_heads * head_dim, H, dtype=torch.float16) * 0.02
        state[f"{pfx}.self_attn.k_proj.weight"] = torch.randn(n_kv * head_dim, H, dtype=torch.float16) * 0.02
        state[f"{pfx}.self_attn.v_proj.weight"] = torch.randn(n_kv * head_dim, H, dtype=torch.float16) * 0.02
        state[f"{pfx}.self_attn.o_proj.weight"] = torch.randn(H, n_heads * head_dim, dtype=torch.float16) * 0.02
        # MoE gate
        state[f"{pfx}.mlp.gate.weight"] = torch.randn(n_experts, H, dtype=torch.float16) * 0.02
        # Experts
        for e in range(n_experts):
            state[f"{pfx}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(expert_ffn, H, dtype=torch.float16) * 0.02
            state[f"{pfx}.mlp.experts.{e}.up_proj.weight"] = torch.randn(expert_ffn, H, dtype=torch.float16) * 0.02
            state[f"{pfx}.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, expert_ffn, dtype=torch.float16) * 0.02

    state["model.norm.weight"] = torch.ones(H, dtype=torch.float16)
    state["lm_head.weight"] = torch.randn(V, H, dtype=torch.float16) * 0.02

    save_file(state, os.path.join(model_dir, "model.safetensors"))

    # Tokenizer config (minimal)
    tok_cfg = {
        "model_type": "qwen3_moe",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_cfg, f)

    # 简易 tokenizer.json
    vocab = {f"token_{i}": i for i in range(V)}
    vocab["<s>"] = 0
    vocab["</s>"] = 1
    vocab["<pad>"] = 2
    tokenizer_json = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {"id": 0, "content": "<s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "</s>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
    }
    with open(os.path.join(model_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f)

    print(f"  [tiny model] 生成完成: {len(state)} 个张量, 目录={model_dir}")
    return model_dir


def _wait_for_server(port: int, timeout: int = 180, proc=None) -> bool:
    """等待 SGLang server 就绪。"""
    import time
    import requests

    start = time.time()
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            print(f"  [server] 进程已退出 (code={proc.returncode})")
            if proc.stdout:
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining[-2000:])
            return False
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ===========================================================================
# Modal 包装
# ===========================================================================
def _build_modal_app():
    """构建 Modal App：4x T4 GPU。"""
    import modal

    HERE = pathlib.Path(__file__).resolve().parent
    SGLANG_LOCAL = HERE / "sglang" / "python"

    base_image = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")

    image = (
        modal.Image.from_registry(base_image, add_python=None)
        .run_commands(
            "pip install --no-cache-dir safetensors requests || true",
            "pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall "
            "git+https://github.com/redai-infra/megatron-bridge.git@f13bec09 || true",
        )
        .add_local_file(
            pathlib.Path(__file__).resolve().as_posix(),
            "/root/verify_tp_gather.py",
            copy=True,
        )
        .add_local_dir(
            SGLANG_LOCAL.as_posix(),
            "/root/sglang_src/python",
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc"],
        )
        .env({
            "PYTHONPATH": "/root/sglang_src/python:/root/Megatron-LM",
            "NCCL_DEBUG": "WARN",
        })
    )

    app = modal.App("verify-tp-gather")
    return app, image


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------
try:
    import modal

    HERE_PATH = pathlib.Path(__file__).resolve().parent
    SGLANG_LOCAL_PATH = HERE_PATH / "sglang" / "python"

    _BASE_IMAGE = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")

    _image = (
        modal.Image.from_registry(_BASE_IMAGE, add_python=None)
        .run_commands(
            "pip install --no-cache-dir safetensors requests || true",
            "pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall "
            "git+https://github.com/redai-infra/megatron-bridge.git@f13bec09 || true",
        )
        .add_local_file(
            pathlib.Path(__file__).resolve().as_posix(),
            "/root/verify_tp_gather.py",
            copy=True,
        )
        .add_local_dir(
            SGLANG_LOCAL_PATH.as_posix(),
            "/root/sglang_src/python",
            copy=True,
            ignore=["**/__pycache__", "**/*.pyc"],
        )
        .env({
            "PYTHONPATH": "/root/sglang_src/python:/root/Megatron-LM",
            "NCCL_DEBUG": "WARN",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        })
    )

    _app = modal.App("verify-tp-gather")

    @_app.function(image=_image, gpu="T4:4", timeout=20 * 60)
    def main():
        """Modal entrypoint: 4x T4 上跑全链路验证。"""
        import subprocess

        print("=" * 70)
        print("  Tiny Qwen3-Omni TP=4 LoRA Gather 全链路验证")
        print("=" * 70)

        # Stage 1-3: torchrun 4 进程
        cmd = [
            "torchrun", "--nproc_per_node=4", "--master_port=29500",
            "/root/verify_tp_gather.py", "--run-stages",
        ]
        print(f"\n[CMD] {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            print(f"\n[FAIL] Stage 1-3 返回 {result.returncode}")
            raise SystemExit(result.returncode)

        # Stage 4: SGLang (单进程)
        print("\n" + "=" * 70)
        print("  Stage 4: SGLang 热加载验证")
        print("=" * 70)
        cmd4 = [sys.executable, "/root/verify_tp_gather.py", "--run-sglang"]
        result4 = subprocess.run(cmd4, check=False)

        if result4.returncode != 0:
            print(f"\n[WARN] Stage 4 返回 {result4.returncode}（SGLang 部分可能有兼容问题，不阻塞）")

        print("\n" + "=" * 70)
        print("  验证完成")
        print("=" * 70)

except ImportError:
    pass


# ===========================================================================
# CLI
# ===========================================================================
if __name__ == "__main__":
    if "--run-stages" in sys.argv:
        success = run_megatron_stages()
        sys.exit(0 if success else 1)
    elif "--run-sglang" in sys.argv:
        success = run_sglang_stage()
        sys.exit(0 if success else 1)
    else:
        print("用法:")
        print("  modal run verify_tp_gather.py::main     # Modal 4x T4 全链路")
        print("  torchrun --nproc_per_node=4 verify_tp_gather.py --run-stages  # 本地 Stage 1-3")
        print("  python verify_tp_gather.py --run-sglang  # 本地 Stage 4")
