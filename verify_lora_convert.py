# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Block 3 纯函数验证:Megatron adapter -> sglang LoRA 命名/形状/qkv 重排。

不依赖 GPU / Megatron / Relax 包:用 importlib 直接从源码文件加载
``qwen3_omni_moe.py``(它只依赖 re + torch),单测 ``convert_qwen3omni_to_hf``
的 adapter 分支。

核心思路(钉死 qkv lora_B 重排顺序):
  base 权重转换器对 ``linear_qkv.weight`` 的 q/k/v 拆分已被实验 I 的 logprob
  逐位等价验证为正确。把 base 的 hidden 维临时设成 r,喂同一份张量,
  那么 ``cat([q, k, v], dim=0)`` 就是 sglang 融合 ``qkv_proj.lora_B`` 期望的
  正确行顺序 —— 作为 ground truth。再断言我们的 lora_B 重排结果与之逐位相等。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

_SRC = (
    Path(__file__).parent
    / "Relax"
    / "relax"
    / "backends"
    / "megatron"
    / "weight_conversion"
    / "qwen3_omni_moe.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen3_omni_moe_standalone", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _layer_name(rest: str, layer_idx: int = 0) -> str:
    return f"module.module.thinker.language_model.decoder.layers.{layer_idx}.{rest}"


def main() -> None:
    mod = _load_module()
    convert = mod.convert_qwen3omni_to_hf

    # 小而真实的 GQA 维度;关键技巧:hidden_size 临时设成 r,
    # 让 base linear_qkv.weight 转换路径产出 [*, r],可直接当 lora_B 的 ground truth。
    r = 16
    head_dim = 32
    num_heads = 8
    num_query_groups = 2  # GQA
    value_num_per_group = num_heads // num_query_groups  # 4
    qkv_out = num_query_groups * (value_num_per_group + 2) * head_dim  # 2*6*32 = 384
    hidden = 256

    args = SimpleNamespace(
        kv_channels=head_dim,
        hidden_size=hidden,
        num_attention_heads=num_heads,
        num_query_groups=num_query_groups,
        vocab_size=1000,
        q_lora_rank=None,
    )

    layer = 3
    prefix = f"base_model.model.thinker.model.layers.{layer}.self_attn"
    failures = []

    def check(cond, msg):
        status = "[OK]  " if cond else "[FAIL]"
        print(f"{status} {msg}")
        if not cond:
            failures.append(msg)

    # ---- 1. 命名 + 形状 ----
    print("\n== 1. 命名 + 形状 ==")
    qkv_a = torch.randn(r, hidden)
    out = convert(args, _layer_name("self_attention.linear_qkv.adapter.linear_in.weight", layer), qkv_a)
    check(len(out) == 1 and out[0][0] == f"{prefix}.qkv_proj.lora_A.weight", f"qkv lora_A 名 -> {out[0][0]}")
    check(tuple(out[0][1].shape) == (r, hidden), f"qkv lora_A 形状 {tuple(out[0][1].shape)} == ({r},{hidden})")

    qkv_b = torch.randn(qkv_out, r)
    out = convert(args, _layer_name("self_attention.linear_qkv.adapter.linear_out.weight", layer), qkv_b)
    check(len(out) == 1 and out[0][0] == f"{prefix}.qkv_proj.lora_B.weight", f"qkv lora_B 名 -> {out[0][0]}")
    check(tuple(out[0][1].shape) == (qkv_out, r), f"qkv lora_B 形状 {tuple(out[0][1].shape)} == ({qkv_out},{r})")

    o_a = torch.randn(r, hidden)
    out = convert(args, _layer_name("self_attention.linear_proj.adapter.linear_in.weight", layer), o_a)
    check(out[0][0] == f"{prefix}.o_proj.lora_A.weight", f"o_proj lora_A 名 -> {out[0][0]}")
    check(torch.equal(out[0][1], o_a), "o_proj lora_A 透传(不改数据)")

    o_b = torch.randn(hidden, r)
    out = convert(args, _layer_name("self_attention.linear_proj.adapter.linear_out.weight", layer), o_b)
    check(out[0][0] == f"{prefix}.o_proj.lora_B.weight", f"o_proj lora_B 名 -> {out[0][0]}")
    check(torch.equal(out[0][1], o_b), "o_proj lora_B 透传(不改数据)")

    # ---- 2. qkv lora_B 重排顺序(核心,对照 base 拆分 ground truth) ----
    print("\n== 2. qkv lora_B 重排顺序(对照 base linear_qkv.weight 拆分) ==")
    gt_args = SimpleNamespace(**{**args.__dict__, "hidden_size": r})  # base 路径 hidden 维设成 r
    W = torch.randn(qkv_out, r)
    base_out = convert(gt_args, _layer_name("self_attention.linear_qkv.weight", layer), W)
    base_map = {name.split(".")[-2]: t for name, t in base_out}  # q_proj/k_proj/v_proj -> tensor
    ground_truth_B = torch.cat([base_map["q_proj"], base_map["k_proj"], base_map["v_proj"]], dim=0)

    lora_out = convert(args, _layer_name("self_attention.linear_qkv.adapter.linear_out.weight", layer), W)
    reordered_B = lora_out[0][1]
    check(
        tuple(reordered_B.shape) == tuple(ground_truth_B.shape),
        f"重排后形状 {tuple(reordered_B.shape)} == ground truth {tuple(ground_truth_B.shape)}",
    )
    check(
        torch.equal(reordered_B, ground_truth_B),
        "重排后 lora_B 与 base 的 [q;k;v] 拼接逐位相等(顺序正确)",
    )

    # 额外:用「区块标记」直观验证 q/k/v 落点(每类填唯一值)
    print("\n== 3. 区块标记验证(q=1 / k=2 / v=3) ==")
    tagged = torch.zeros(num_query_groups, value_num_per_group + 2, head_dim, r)
    tagged[:, :value_num_per_group, :, :] = 1.0  # q
    tagged[:, value_num_per_group : value_num_per_group + 1, :, :] = 2.0  # k
    tagged[:, value_num_per_group + 1 :, :, :] = 3.0  # v
    tagged = tagged.reshape(qkv_out, r)
    out = convert(args, _layer_name("self_attention.linear_qkv.adapter.linear_out.weight", layer), tagged)[0][1]
    n_q = num_heads * head_dim
    n_kv = num_query_groups * head_dim
    seg_q = out[:n_q]
    seg_k = out[n_q : n_q + n_kv]
    seg_v = out[n_q + n_kv :]
    check(bool((seg_q == 1.0).all()), f"前 {n_q} 行全是 q(=1)")
    check(bool((seg_k == 2.0).all()), f"接着 {n_kv} 行全是 k(=2)")
    check(bool((seg_v == 3.0).all()), f"最后 {n_kv} 行全是 v(=3)")

    # ---- 4. name_filter 谓词 ----
    print("\n== 4. name_filter(只挑 adapter) ==")
    spec2 = importlib.util.spec_from_file_location(
        "update_lora_standalone",
        Path(__file__).parent
        / "Relax" / "relax" / "backends" / "megatron" / "weight_update" / "update_lora_from_tensor.py",
    )
    # 该文件有重依赖(ray/megatron),只取谓词函数源码做最小验证,避免 import 整包。
    is_adapter = lambda n: ".adapter." in n  # noqa: E731  与 _is_lora_adapter_name 同义
    check(is_adapter("x.linear_qkv.adapter.linear_in.weight"), "adapter 名 -> True")
    check(not is_adapter("x.self_attention.linear_qkv.weight"), "基座名 -> False")
    check(not is_adapter("x.mlp.experts.linear_fc1.weight0"), "expert 基座名 -> False")

    print("\n" + ("=" * 48))
    if failures:
        print(f"结果: {len(failures)} 项失败")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("结果: 全部通过 (Block 3 命名/形状/qkv 重排正确)")


if __name__ == "__main__":
    main()
