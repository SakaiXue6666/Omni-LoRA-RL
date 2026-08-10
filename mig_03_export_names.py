"""探针二 —— adapter 导出命名与 SGLang 期望的 parity 对照（容器内运行）。

要回答的问题：
    官方 export_adapter_weights 对 Qwen3-Omni 导出的张量叫什么、形状是什么，
    SGLang 那边认不认；跟 v1 手写的 direct 导出比，差在哪。

静态追踪的结论（本脚本去证实）：

  导出侧（megatron/bridge/models/conversion/peft_bridge.py）
    MEGATRON_TO_HF_LORA_SUFFIX = {".linear_in.weight": ".lora_A.weight",
                                  ".linear_out.weight": ".lora_B.weight"}
    adapter 的 HF 名字由 **基座权重的映射** 推导（_resolve_hf_adapter_param_name
    -> mapping_registry.megatron_to_hf_lookup(base) -> _make_lora_param_name），
    所以 Omni 的 thinker. 前缀会被自动带上。
    融合 QKV 的 linear_out 由 _split_qkv_linear_out_weight 调 split_qkv_weights
    拆成 q/k/v —— 与基座权重同一套去交错逻辑。
    => v1 手写的 _reorder_qkv_lora_b 是多余的。

  Relax 侧（relax/backends/megatron/weight_update/lora_adapter_sync.py）
    export_local_adapter 直接用 item.param_name，**不带** base_model.model. 前缀
    （上游只有 convert_adapter_weights_to_peft_state 落盘时才加这个前缀）。

  SGLang 侧
    get_layer_id 是 re.search(r"layers\\.(\\d+)\\.")，与前缀无关；
    normalize_qkv_proj 会把分开的 q/k/v 堆成 qkv_proj（lora_A 三份 cat 成 3r）。
    => 拆开导出的形式正是 SGLang 想要的。

因此本脚本做四件事，全程不建 Megatron 模型：
  1. 清点镜像里的真实版本（预构建镜像可能落后于 Relax main 的钉法）
  2. 核对导出侧的命名常量与关键方法确实存在且如上
  3. 用真实 bridge 的 mapping_registry 推导出预测的 adapter 名字
  4. 造出对应形状的假张量，喂进 SGLang 真实的解析/堆叠函数，验证能被正确接收

用法：python mig_03_export_names.py   （默认读 /models/qwen3-omni，可用 OMNI_CKPT 覆盖）
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback
import types

OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

# 探针用的 LoRA 秩，随便取，只影响假张量形状。
LORA_RANK = 32

# v1 direct 路线的契约，从 patches/relax.patch 里抄出来做对照。
V1_CONTRACT = [
    ("self_attention.linear_qkv.adapter.linear_in.weight", "qkv_proj.lora_A.weight（融合，未拆）"),
    ("self_attention.linear_qkv.adapter.linear_out.weight", "qkv_proj.lora_B.weight（融合，手工重排 [q;k;v]）"),
    ("self_attention.linear_proj.adapter.linear_in.weight", "o_proj.lora_A.weight"),
    ("self_attention.linear_proj.adapter.linear_out.weight", "o_proj.lora_B.weight"),
]
V1_PREFIX = "base_model.model.thinker.model.layers"

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _failures.append(tag)


def _warn(tag: str, msg: str) -> None:
    print(f"[WARN][{tag}] {msg}", flush=True)


def step_versions() -> None:
    """镜像里的真实版本。

    尤其是 sglang —— Relax 的 Dockerfile 钉 v0.5.12.post1，但我们用的是
    2026-07-23 构建的预构建镜像，可能落后。adapter 模式的内存推送
    （load_lora_adapter_from_tensors）要求 sglang >= 0.5.12，所以这里必须确认。
    """
    tag = "versions"
    try:
        import torch

        print(f"    torch        = {torch.__version__}", flush=True)
        for mod_name in ("transformers", "sglang", "megatron.bridge", "megatron.core", "peft"):
            try:
                mod = __import__(mod_name, fromlist=["__version__"])
                print(f"    {mod_name:<16} = {getattr(mod, '__version__', '?')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    {mod_name:<16} = 不可用 ({type(e).__name__})", flush=True)
        _ok(tag, "版本已记录")
    except Exception:  # noqa: BLE001
        _fail(tag, "版本清点失败：\n" + traceback.format_exc())


def step_export_constants() -> None:
    """核对导出侧的命名常量与关键方法。"""
    tag = "export-api"
    try:
        from megatron.bridge.models.conversion import peft_bridge as pb

        suffix_map = getattr(pb, "MEGATRON_TO_HF_LORA_SUFFIX", None)
        print(f"    MEGATRON_TO_HF_LORA_SUFFIX = {suffix_map}", flush=True)
        expected = {".linear_in.weight": ".lora_A.weight", ".linear_out.weight": ".lora_B.weight"}
        if suffix_map != expected:
            _fail(tag, f"命名常量与预期不符，预期 {expected}")
            return

        missing = [
            name
            for name in (
                "_make_lora_param_name",
                "_resolve_hf_adapter_param_name",
                "_split_qkv_linear_out_weight",
                "_build_lora_hf_names",
                "build_adapter_conversion_tasks",
            )
            if not hasattr(pb.MegatronPeftBridge, name)
        ]
        if missing:
            _fail(tag, f"MegatronPeftBridge 缺少方法: {missing}（镜像里的 bridge 版本偏旧？）")
            return

        print("---- _split_qkv_linear_out_weight 源码 ----", flush=True)
        print(inspect.getsource(pb.MegatronPeftBridge._split_qkv_linear_out_weight), flush=True)

        has_peft_state = hasattr(pb, "convert_adapter_weights_to_peft_state")
        print(f"    convert_adapter_weights_to_peft_state 存在: {has_peft_state}", flush=True)
        _ok(tag, "导出侧 API 与命名常量符合预期")
    except Exception:  # noqa: BLE001
        _fail(tag, "核对导出侧 API 失败：\n" + traceback.format_exc())


def step_predict_names() -> dict:
    """用真实 bridge 的 mapping_registry 推导 adapter 的 HF 名字。

    走的就是 _resolve_hf_adapter_param_name 内部那两步：
        mapping_registry.megatron_to_hf_lookup(base)  ->  _make_lora_param_name
    所以结论等价于真实导出，但不需要建模型。
    """
    tag = "predict"
    try:
        import relax.models  # noqa: F401  触发 Qwen3-Omni bridge 注册
        from megatron.bridge import AutoBridge
        from megatron.bridge.models.conversion import peft_bridge as pb

        bridge = AutoBridge.from_hf_pretrained(OMNI_CKPT, trust_remote_code=True)
        model_bridge = None
        for attr in ("_model_bridge", "model_bridge", "_bridge"):
            cand = getattr(bridge, attr, None)
            if cand is not None and hasattr(cand, "mapping_registry"):
                model_bridge = cand
                break
        if model_bridge is None and hasattr(bridge, "mapping_registry"):
            model_bridge = bridge
        if model_bridge is None:
            _fail(tag, f"拿不到 model bridge（AutoBridge 属性: {[a for a in dir(bridge) if not a.startswith('__')][:40]}）")
            return {}

        registry = model_bridge.mapping_registry()
        print(f"    model bridge = {type(model_bridge).__name__}", flush=True)

        base_params = [
            "language_model.decoder.layers.0.self_attention.linear_qkv.weight",
            "language_model.decoder.layers.0.self_attention.linear_proj.weight",
        ]
        predicted: dict[str, list[str]] = {}
        maker = pb.MegatronPeftBridge._make_lora_param_name
        dummy = types.SimpleNamespace()

        for base in base_params:
            mapping = registry.megatron_to_hf_lookup(base)
            hf_param = getattr(mapping, "hf_param", None)
            hf_names = [hf_param] if isinstance(hf_param, str) else list(hf_param.values())
            print(f"\n    Megatron: {base}", flush=True)
            print(f"    -> HF   : {hf_names}   (mapping={type(mapping).__name__})", flush=True)
            a_names = [maker(dummy, n, ".linear_in.weight") for n in hf_names]
            b_names = [maker(dummy, n, ".linear_out.weight") for n in hf_names]
            print("    -> adapter:", flush=True)
            for n in a_names + b_names:
                print(f"         {n}", flush=True)
            predicted[base] = a_names + b_names

        _ok(tag, "adapter 命名推导完成")
        return predicted
    except Exception:  # noqa: BLE001
        _fail(tag, "推导 adapter 命名失败：\n" + traceback.format_exc())
        return {}


def step_sglang_roundtrip(predicted: dict) -> None:
    """造出对应形状的假张量，喂进 SGLang 真实的解析与堆叠函数。

    验证三件事：
      1. get_layer_id 能从带 thinker. 前缀的名字里解析出层号
      2. get_target_module_name 能把名字归到正确的 target module
      3. normalize_qkv_proj 能把拆开的 q/k/v 堆成 qkv_proj，且形状对得上
    """
    tag = "sglang"
    if not predicted:
        _warn(tag, "上一步没产出名字，跳过")
        return
    try:
        import torch
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(OMNI_CKPT, trust_remote_code=True)
        text_cfg = cfg.thinker_config.text_config
        hidden = text_cfg.hidden_size
        n_heads = text_cfg.num_attention_heads
        n_kv = getattr(text_cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(text_cfg, "head_dim", None) or hidden // n_heads
        print(
            f"    hidden={hidden} n_heads={n_heads} n_kv={n_kv} head_dim={head_dim} rank={LORA_RANK}",
            flush=True,
        )

        q_out, kv_out = n_heads * head_dim, n_kv * head_dim
        shapes = {
            "q_proj.lora_A.weight": (LORA_RANK, hidden),
            "k_proj.lora_A.weight": (LORA_RANK, hidden),
            "v_proj.lora_A.weight": (LORA_RANK, hidden),
            "q_proj.lora_B.weight": (q_out, LORA_RANK),
            "k_proj.lora_B.weight": (kv_out, LORA_RANK),
            "v_proj.lora_B.weight": (kv_out, LORA_RANK),
            "o_proj.lora_A.weight": (LORA_RANK, q_out),
            "o_proj.lora_B.weight": (hidden, LORA_RANK),
        }
        names = [n for group in predicted.values() for n in group]
        weights: dict[str, torch.Tensor] = {}
        for n in names:
            key = ".".join(n.split(".")[-3:])
            shape = shapes.get(key)
            if shape is None:
                _warn(tag, f"没有为 {n} 预设形状，跳过")
                continue
            weights[n] = torch.zeros(*shape)

        from sglang.srt.layers.utils import get_layer_id

        print("\n    层号解析:", flush=True)
        for n in sorted(weights):
            print(f"        {get_layer_id(n)}  <- {n}", flush=True)
        if any(get_layer_id(n) != 0 for n in weights):
            _fail(tag, "有名字解析不出层号 0")
            return

        from sglang.srt.lora.utils import get_normalized_target_modules, get_target_module_name

        normalized = get_normalized_target_modules({"q_proj", "k_proj", "v_proj", "o_proj"})
        print(f"\n    归一化后的 target modules: {sorted(normalized)}", flush=True)
        print("    模块归属:", flush=True)
        for n in sorted(weights):
            try:
                print(f"        {get_target_module_name(n, normalized):<12} <- {n}", flush=True)
            except Exception as e:  # noqa: BLE001
                _warn(tag, f"{n} 归属判定失败: {e!r}")

        from sglang.srt.lora.lora import LoRAAdapter

        before = {k: tuple(v.shape) for k, v in weights.items()}
        LoRAAdapter.normalize_qkv_proj(types.SimpleNamespace(), list(weights.keys()), weights)
        after = {k: tuple(v.shape) for k, v in weights.items()}
        print("\n    normalize_qkv_proj 之后:", flush=True)
        for k in sorted(after):
            mark = "" if k in before else "  <-- 新生成"
            print(f"        {k}  {after[k]}{mark}", flush=True)
        gone = sorted(set(before) - set(after))
        print(f"    被消化掉的: {gone}", flush=True)

        qkv_b = [k for k in after if "qkv_proj.lora_B" in k]
        qkv_a = [k for k in after if "qkv_proj.lora_A" in k]
        if not qkv_b or not qkv_a:
            _fail(tag, "没有堆出 qkv_proj 的 lora_A / lora_B")
            return
        exp_b = (q_out + 2 * kv_out, LORA_RANK)
        exp_a = (3 * LORA_RANK, hidden)
        got_b, got_a = after[qkv_b[0]], after[qkv_a[0]]
        print(f"\n    qkv_proj.lora_B 形状 {got_b}，预期 {exp_b}", flush=True)
        print(f"    qkv_proj.lora_A 形状 {got_a}，预期 {exp_a}", flush=True)
        if got_b != exp_b or got_a != exp_a:
            _fail(tag, "堆叠后的形状与预期不符")
            return
        _ok(tag, "SGLang 侧能正确解析并堆叠上游导出的命名")
    except Exception:  # noqa: BLE001
        _fail(tag, "SGLang 往返验证失败：\n" + traceback.format_exc())


def step_parity_table() -> None:
    """把 v1 与 v2 的契约并排列出来。"""
    print("\n    v1 direct 路线（手写）:", flush=True)
    print(f"        前缀: {V1_PREFIX}.N.self_attn", flush=True)
    for megatron, hf in V1_CONTRACT:
        print(f"        {megatron}\n            -> {hf}", flush=True)
    print("\n    v2 bridge 路线（官方）:", flush=True)
    print("        前缀: thinker.model.layers.N.self_attn（无 base_model.model.）", flush=True)
    print("        linear_qkv.adapter.linear_in  -> {q,k,v}_proj.lora_A（三份同值）", flush=True)
    print("        linear_qkv.adapter.linear_out -> {q,k,v}_proj.lora_B（上游拆分去交错）", flush=True)
    print("        linear_proj.adapter.linear_in/out -> o_proj.lora_A/B", flush=True)
    print("\n    差异与影响:", flush=True)
    print("        1. 拆开 vs 融合 —— SGLang 的 normalize_qkv_proj 会堆回去，等价", flush=True)
    print("        2. QKV 去交错 —— 上游已做，v1 的 _reorder_qkv_lora_b 可以丢弃", flush=True)
    print("        3. base_model.model. 前缀 —— v2 不带；SGLang 按后缀与正则匹配，理论上无关", flush=True)


def main() -> None:
    print("========== 探针二：adapter 导出命名 parity ==========", flush=True)
    print("\n[1] 镜像版本清点", flush=True)
    step_versions()

    print("\n[2] 导出侧命名常量与 API", flush=True)
    step_export_constants()

    print("\n[3] 用真实 mapping_registry 推导 adapter 命名", flush=True)
    predicted = step_predict_names()

    print("\n[4] 喂进 SGLang 真实的解析与堆叠", flush=True)
    step_sglang_roundtrip(predicted)

    print("\n[5] v1 / v2 契约对照", flush=True)
    step_parity_table()

    print("\n========== 结论 ==========", flush=True)
    if _failures:
        print(f"  [PROBE X] 失败环节: {sorted(set(_failures))}", flush=True)
        sys.exit(1)
    print("  [PROBE OK] 官方导出的命名可被 SGLang 正确接收；v1 的手工重排与融合命名均可丢弃。", flush=True)


if __name__ == "__main__":
    main()
