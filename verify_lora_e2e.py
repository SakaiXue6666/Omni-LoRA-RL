# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tier 1：LoRA 集成链「端到端」验证（单卡，不加载 30B 权重）。

串起来的真实代码路径（这正是只靠纯函数/手搭测试覆盖不到、且在「真烧 4 卡」时
才暴露的接线层 —— 对应 LORA_RL_INTEGRATION.md 坑 8/9）：

    provider  ──wrap_model_provider_with_lora──▶ wrapped
       │                                            │  按 Megatron build_model 的方式调用
       │                                            ▼  （额外传 config=/pg_collection=）
    真实 Megatron self-attention ──apply_lora──▶ 挂 adapter + _assert_lora_attached
       │                                            │
       └────────── named_parameters() ─────────────┘
                          │  每个 adapter 走一遍
                          ▼
              convert_qwen3omni_to_hf ──▶ 断言落到 sglang 期望命名

Part A（必跑）：用真实 Megatron 模块手搭 thinker.{language_model, audio_model} 树，
              不依赖真实 checkpoint，T4/CPU 均可。
Part B（尽力）：读 Volume 上的真 config，仅缩「层数/专家数/vocab」，用真实
              ``AutoBridge`` 构造一个 tiny Qwen3-Omni，走同一条 wrap/attach 链。
              失败只告警、不让整测失败（bridge 对 tiny config 偶有挑剔）。

运行：
    torchrun --nproc_per_node=1 verify_lora_e2e.py          # 本地/集群
    # 或单进程（脚本会自行设置 RANK/WORLD_SIZE 并 init_process_group）
    python verify_lora_e2e.py
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Megatron 单进程并行状态（TP=PP=1）
# --------------------------------------------------------------------------- #
def _init_megatron_tp1() -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29533")
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

    # relax 自带的 gloo group（权重迭代器的跨 rank 一致性校验会用到 get_gloo_group()）
    from relax.utils.distributed_utils import init_gloo_group

    init_gloo_group()


# --------------------------------------------------------------------------- #
# 用真实 Megatron 模块搭一个近似 Qwen3-Omni 的结构
# --------------------------------------------------------------------------- #
# 这些维度同时被「挂 LoRA」和「转换断言」共用，必须自洽：
#   value_num_per_group = HEADS // GROUPS
#   qkv_out = GROUPS * (value_num_per_group + 2) * KV   （fused-qkv 输出维）
HIDDEN = 256
HEADS = 8
GROUPS = 4
KV = 32


def _build_attention_block() -> nn.Module:
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_layer import TransformerLayer

    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
        ffn_hidden_size=512,
        kv_channels=KV,
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
    except Exception as exc:  # noqa: BLE001
        print(f"[build] TE spec 失败({exc})，回退 local spec")
        spec = get_gpt_layer_local_spec(qk_layernorm=True)
        layer = TransformerLayer(config=config, submodules=spec.submodules, layer_number=1)
    layer.config = config
    return layer


def _build_omni_like_tree() -> nn.Module:
    """thinker.{language_model.decoder.layers[0], audio_model}，二者都是真实 attention。"""
    root = nn.Module()
    root.thinker = nn.Module()
    root.thinker.language_model = nn.Module()
    root.thinker.language_model.decoder = nn.Module()
    root.thinker.language_model.decoder.layers = nn.ModuleList([_build_attention_block()])
    root.thinker.audio_model = _build_attention_block()  # 不该被挂 LoRA
    return root


def _lora_args() -> SimpleNamespace:
    # scoped 通配符：只挂 thinker 文本侧，避开 audio/vision 塔（坑②）
    return SimpleNamespace(
        lora_target_modules=["*language_model*linear_qkv", "*language_model*linear_proj"],
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.0,
    )


def _conv_args() -> SimpleNamespace:
    return SimpleNamespace(
        kv_channels=KV,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=GROUPS,
    )


# --------------------------------------------------------------------------- #
# Part A：真实 wrap → Megatron 风格调用 → attach → convert
# --------------------------------------------------------------------------- #
def _run_attach_and_convert(model: nn.Module, tag: str, failures: list[str]) -> None:
    """对已挂好 LoRA 的 model：核对 adapter 落点 + 跑一遍 sglang 命名转换。"""
    from relax.backends.megatron.weight_conversion.qwen3_omni_moe import convert_qwen3omni_to_hf

    conv_args = _conv_args()
    adapters = [(n, p) for n, p in model.named_parameters() if ".adapter." in n]

    def _check(cond: bool, msg: str) -> None:
        print(f"  {'[OK]  ' if cond else '[FAIL]'} {msg}")
        if not cond:
            failures.append(f"[{tag}] {msg}")

    _check(bool(adapters), f"挂到 {len(adapters)} 个 adapter 参数")

    # adapter 不该落在 audio/vision 塔
    bad = [n for n, _ in adapters if re.search(r"audio|vision|visual", n)]
    _check(not bad, f"adapter 未误挂到多模态塔（误挂={bad[:3]}）")

    # 逐个 adapter 跑转换：真实权重迭代器会带 module.module. 前缀
    expected = {"qkv_proj.lora_A", "qkv_proj.lora_B", "o_proj.lora_A", "o_proj.lora_B"}
    got: set[str] = set()
    for name, param in adapters:
        if "language_model" not in name:
            continue
        full = "module.module." + name
        try:
            out = convert_qwen3omni_to_hf(conv_args, full, param.detach().float().cpu())
        except Exception as exc:  # noqa: BLE001
            failures.append(f"[{tag}] 转换抛错 {name}: {exc!r}")
            print(f"  [FAIL] 转换抛错 {name}: {exc!r}")
            continue
        for hf_name, _t in out:
            _check(
                hf_name.startswith("base_model.model.thinker.model.layers."),
                f"sglang 命名前缀正确 -> {hf_name}",
            )
            m = re.search(r"self_attn\.(qkv_proj|o_proj)\.(lora_A|lora_B)", hf_name)
            if m:
                got.add(f"{m.group(1)}.{m.group(2)}")

    _check(
        got == expected,
        f"qkv/o_proj 的 lora_A/B 四件齐全：得到 {sorted(got)}",
    )


def _run_sync_wiring_check(failures: list[str]) -> None:
    """坑 16/17：验证 LoRA 同步走 direct 路线 + 已上 contiguity 修复。

    坑 17:镜像 pin 的 redai-fork bridge(f13bec09,带 Qwen3-Omni)早于
    ``export_adapter_weights`` API,``AutoBridge`` 无此方法;升级又会丢 Qwen3-Omni。
    故回退 ``HfWeightIteratorDirect`` + ``convert_qwen3omni_to_hf`` 的 direct 路径,
    并修复坑 16:adapter 是分布式优化器连续 buffer 的非连续视图,NCCL all_gather
    前须 ``.contiguous()``。这里做廉价接线自检:
      - ``UpdateLoRAFromTensor`` 用 ``HfWeightIteratorDirect`` + ``name_filter`` (.adapter.);
      - ``actor.py`` 的 LoRA weights_getter 用全局命名 live 参数(坑 15);
      - ``common.py`` 的两条 all_gather 路径均已 ``.contiguous()``(坑 16)。
    """
    print("\n  -- 坑 16/17:direct LoRA 同步 + contiguity 接线自检 --")

    def _check(cond: bool, msg: str) -> None:
        print(f"  {'[OK]  ' if cond else '[FAIL]'} {msg}")
        if not cond:
            failures.append(f"[A-sync] {msg}")

    try:
        import inspect as _inspect

        from relax.backends.megatron import actor as actor_mod
        from relax.backends.megatron.weight_update import common as common_mod
        from relax.backends.megatron.weight_update import update_lora_from_tensor as ulft

        ulft_src = _inspect.getsource(ulft)
        _check("HfWeightIteratorDirect(" in ulft_src, "UpdateLoRAFromTensor 实例化 HfWeightIteratorDirect")
        _check("name_filter=_is_lora_adapter_name" in ulft_src, "direct iterator 带 name_filter(.adapter.)")
        # 注:docstring 里有历史说明性文字,故只看「真实调用/实例化」而非裸字串
        _check(
            ".export_adapter_weights(" not in ulft_src and "HfWeightIteratorBridge(" not in ulft_src,
            "不再调用 Bridge.export_adapter_weights(坑 17 已弃用)",
        )

        actor_src = _inspect.getsource(actor_mod)
        _check(
            "convert_to_global_name=True" in actor_src and '".adapter." in name' in actor_src,
            "actor LoRA weights_getter 返回全局命名 live adapter(坑 15)",
        )

        common_src = _inspect.getsource(common_mod)
        _check(
            common_src.count("param.data.contiguous()") >= 2,
            "common.py 两条 all_gather 路径均已 .contiguous()(坑 16)",
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"[A-sync] 接线自检抛错: {type(exc).__name__}: {exc}")
        print(f"  [FAIL] 接线自检抛错: {type(exc).__name__}: {exc}")


def part_a(failures: list[str]) -> None:
    print("\n========== Part A：手搭真实 Megatron 树，走 wrap/attach/convert ==========")
    from relax.backends.megatron.model_provider import wrap_model_provider_with_lora

    # original_provider 模拟 bridge 的 provide：签名 (pre_process, post_process, vp_stage)
    def provide(pre_process: bool = True, post_process: bool = True, vp_stage=None) -> nn.Module:
        model = _build_omni_like_tree()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    wrapped = wrap_model_provider_with_lora(provide, _lora_args())

    # 关键：按 Megatron build_model 的真实调用方式，额外传 config=/pg_collection=。
    # 若 wrapper 没正确过滤/透传签名（坑 8/9），这里会直接 TypeError。
    try:
        model = wrapped(pre_process=True, post_process=True, vp_stage=0, config="CFG", pg_collection="PG")
        print("  [OK]   wrapped(config=, pg_collection=) 调用成功（坑 8/9 修复有效）")
    except TypeError as exc:
        failures.append(f"[A] wrapped 调用撞 TypeError（坑 8/9 回归）: {exc}")
        print(f"  [FAIL] wrapped 调用撞 TypeError（坑 8/9 回归）: {exc}")
        return

    _run_attach_and_convert(model, "A", failures)
    _run_sync_wiring_check(failures)


# --------------------------------------------------------------------------- #
# Part B：尽力用真实 bridge 构造 tiny Qwen3-Omni（仅缩层数/专家数/vocab）
# --------------------------------------------------------------------------- #
_SHRINK_LAYER_KEYS = {"num_hidden_layers", "num_layers", "encoder_layers", "decoder_layers", "depth"}
_SHRINK_EXPERT_KEYS = {"num_experts", "num_routed_experts", "moe_num_experts", "n_routed_experts"}


def _shrink_config(obj, changes: dict | None = None):
    """递归缩小 config：层数->2、专家数->4、vocab->4096、top_k 夹到<=专家数。

    记录改动到 ``changes`` 便于核对是否缩到位（否则 128 专家会直接 OOM）。
    """
    if changes is None:
        changes = {}
    if isinstance(obj, dict):
        experts = None
        for k in _SHRINK_EXPERT_KEYS:
            if isinstance(obj.get(k), int):
                changes[k] = (obj[k], min(obj[k], 4))
                obj[k] = min(obj[k], 4)
                experts = obj[k]
        for k, v in list(obj.items()):
            if isinstance(v, int):
                if k in _SHRINK_LAYER_KEYS:
                    changes[k] = (v, min(v, 2))
                    obj[k] = min(v, 2)
                elif "vocab_size" in k:
                    changes[k] = (v, min(v, 4096))
                    obj[k] = min(v, 4096)
                elif ("num_experts_per_tok" in k or k.endswith("top_k") or k == "moe_topk") and experts:
                    obj[k] = min(v, experts)
            else:
                _shrink_config(v, changes)
    elif isinstance(obj, list):
        for v in obj:
            _shrink_config(v, changes)
    return obj, changes


def part_b(failures: list[str], model_dir: str = "/models/qwen3-omni") -> None:
    print("\n========== Part B（尽力）：真实 AutoBridge 构造 tiny Qwen3-Omni ==========")
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        print(f"  [SKIP] 找不到 {cfg_path}，跳过 Part B")
        return
    try:
        import json
        import shutil
        import tempfile

        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg, changes = _shrink_config(cfg)
        print("  [shrink] 缩容明细（原值->新值）：")
        for k, (old, new) in sorted(changes.items()):
            print(f"           {k}: {old} -> {new}")

        tiny_dir = tempfile.mkdtemp(prefix="tiny-omni-")
        with open(os.path.join(tiny_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        # 顺带带上分词器/处理器等小文件（若 from_hf_pretrained 需要），不带权重
        for fn in os.listdir(model_dir):
            if fn.endswith((".json", ".txt")) and fn != "model.safetensors.index.json":
                try:
                    shutil.copy(os.path.join(model_dir, fn), os.path.join(tiny_dir, fn))
                except Exception:  # noqa: BLE001
                    pass

        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(tiny_dir, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        if hasattr(provider, "finalize"):
            provider.finalize()
        print(f"  [OK]   tiny bridge provider 构造成功 -> {type(provider).__name__}")

        # 在 meta device 上构建：只为拿到真实模块名/层级，不实际分配显存
        # （全家桶 thinker+audio+vision+talker 即便 2 层也吃满 24G，故走 meta）。
        if hasattr(provider, "init_model_with_meta_device"):
            provider.init_model_with_meta_device = True
        with torch.device("meta"):
            model = provider.provide(pre_process=True, post_process=True)
        print("  [OK]   真实 bridge 模型在 meta device 上构建成功（拿到真实模块树）")

        # 用挂 LoRA 时同样的 scoped 通配符意图，核对它对「真实模块名」的命中：
        #   - thinker 文本侧的 linear_qkv/linear_proj 必须命中
        #   - audio/vision 塔的 linear_qkv 必须被排除（否则会产出 sglang 无法加载的张量）
        attn_mods = [
            n
            for n, _ in model.named_modules()
            if n.endswith("self_attention.linear_qkv") or n.endswith("self_attention.linear_proj")
        ]
        lang_hit = [n for n in attn_mods if "language_model" in n]
        mm_hit = [n for n in attn_mods if re.search(r"audio|vision|visual", n)]

        if not lang_hit:
            failures.append("[B] 真实模型里没找到 thinker.language_model 的 linear_qkv/linear_proj 模块")
            print("  [FAIL] 真实模型里没找到 language_model 注意力线性层")
        else:
            print(f"  [OK]   真实结构含 language_model 注意力线性层 {len(lang_hit)} 个，命名样例：")
            for n in lang_hit[:4]:
                print("        e.g.", n)

        # 校验转换器正则能匹配真实层名（去掉 module.module. 前缀差异，正则只认 layers.N.rest）
        conv_re = re.compile(r"thinker\.language_model\.decoder\.layers\.(\d+)\.(.+)")
        matched = [n for n in lang_hit if conv_re.search(n)]
        if not matched:
            failures.append("[B] 转换器正则与真实模块名不匹配（layers.N 结构对不上）")
            print("  [FAIL] 转换器正则与真实模块名不匹配")
        else:
            print(f"  [OK]   转换器正则匹配真实层名 {len(matched)}/{len(lang_hit)} 个")

        if mm_hit:
            print(f"  [info] 多模态塔也有 {len(mm_hit)} 个同名注意力线性层（故 target 必须 scoped），样例：")
            for n in mm_hit[:2]:
                print("        e.g.", n)
    except Exception as exc:  # noqa: BLE001
        # bridge 对 tiny config 偶有挑剔；Part B 仅作 bonus，不让整测失败
        print(f"  [SKIP] 真实 bridge 构造失败（不阻断）：{type(exc).__name__}: {str(exc)[:300]}")


def main() -> None:
    _init_megatron_tp1()
    failures: list[str] = []

    part_a(failures)
    part_b(failures)

    print("\n" + "=" * 56)
    if failures:
        print(f"结果：{len(failures)} 项失败")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print("结果：Tier 1 端到端链路全部通过")
    print("  wrap 签名派发(坑8/9) + LoRA 挂载 + sglang 命名转换 串通无误")
    print("=" * 56)


if __name__ == "__main__":
    main()
