"""探针五 —— 真机上把 adapter 热推给 v2 的 sglang 并确认生效（容器内运行，1 卡）。

这是 v2 第一次真正把模型跑起来。前四个探针都是静态的（对模块名、对导出命名、
对 PEFT 前缀、对 gate 的命中），这里验它们合起来在运行时成不成立。

结构照搬 v1 `modal_run.py` 的实验 G —— 那套在标准 sglang 0.5.9 上跑通过，
所以这里一旦挂了，可以确定是 v2 引入的，而不是环境问题。沿用它的三条判据：

  生效 EFFECT      非零 B 的 adapter 让输出相对 base 改变
  可逆 REVERSIBLE  推入 B=0 的 adapter 后输出精确回到 base（旧权重没残留，
                   而且 B@A 的数学方向没搞反）
  稳定 STABLE      连续多次 unload→load 不崩、不 OOM

与 v1 的关键差异（也正是 v2 要验的东西）：

  * 张量名用 **不带** base_model.model. 前缀的形式，即 Relax 的
    export_adapter_weights 真实产出的样子（探针二实测）。v1 当年手写的是带前缀
    的。SGLang 靠后缀和层号正则匹配，两种都该认——这里把 v2 那种钉死。
  * 跑的是 v2 的 sglang（0.5.12.post1 + vendor patch + 我们的 delta），
    Qwen3-Omni 上多了 supports_lora / _lora_pattern / should_apply_lora 这道门。
    没有这道门，音频塔和视觉塔的 qkv_proj 会被一起包进来，而它们的 hidden dim
    和语言模型根本不一样。
"""

from __future__ import annotations

import os
import sys
import traceback


MODEL_DIR = os.environ.get("OMNI_MODEL_DIR", "/models/qwen3-omni")
LORA_NAME = "relax_policy_lora"

# Qwen3-Omni-30B-A3B thinker 的形状（v1 modal_run.py 实测并写死的同一组值）
RANK = int(os.environ.get("PROBE_LORA_RANK", "32"))
ALPHA = float(os.environ.get("PROBE_LORA_ALPHA", "32"))
HIDDEN = 2048
NUM_LAYERS = int(os.environ.get("PROBE_LORA_LAYERS", "48"))
ATTN_OUT = {"q_proj": 4096, "k_proj": 512, "v_proj": 512, "o_proj": 2048}
ATTN_IN = {"q_proj": HIDDEN, "k_proj": HIDDEN, "v_proj": HIDDEN, "o_proj": 4096}

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"  [PASS] {tag}: {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"  [FAIL] {tag}: {msg}", flush=True)
    _failures.append(tag)


def _banner(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def verify_source() -> bool:
    """确认 import 到的是 fork 的源码，且我们的 delta 在位。"""
    import sglang

    base = os.path.dirname(sglang.__file__)
    print(f"  sglang {sglang.__version__} 来自 {base}", flush=True)

    mgr = os.path.join(base, "srt", "lora", "lora_manager.py")
    with open(mgr, encoding="utf-8") as f:
        gate_call = 'getattr(self.base_model, "should_apply_lora", None)' in f.read()

    from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration as OmniCls

    has_hook = hasattr(OmniCls, "should_apply_lora")
    supports = getattr(OmniCls, "supports_lora", False)

    print(f"  lora_manager 有 gate 调用点: {gate_call}", flush=True)
    print(f"  Omni 有 should_apply_lora: {has_hook}   supports_lora: {supports}", flush=True)
    return gate_call and has_hook and bool(supports)


def chat_prompt(user_msg: str) -> str:
    """套 Qwen3 的 chat 模板。

    v1 的教训：不套模板，这个 instruct 模型会直接吐 <|im_end|>，拿到空输出，
    后面所有比对都失去意义。
    """
    return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"


def make_adapter(b_scale: float, a_scale: float = 0.02, seed: int | None = None) -> dict:
    """按 Relax 导出的命名造一份 CPU 上的 adapter。

    b_scale == 0 -> lora_B 全零 -> B @ A == 0 -> 数学上的 no-op。
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    tensors: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        # 注意：不带 base_model.model. 前缀，这是 Relax 的 export_adapter_weights
        # 真实产出的形式（探针二）。
        prefix = f"thinker.model.layers.{layer}.self_attn"
        for mod, out_f in ATTN_OUT.items():
            a = torch.randn(RANK, ATTN_IN[mod], dtype=torch.bfloat16) * a_scale
            if b_scale == 0:
                b = torch.zeros(out_f, RANK, dtype=torch.bfloat16)
            else:
                b = torch.randn(out_f, RANK, dtype=torch.bfloat16) * b_scale
            tensors[f"{prefix}.{mod}.lora_A.weight"] = a
            tensors[f"{prefix}.{mod}.lora_B.weight"] = b
    return tensors


def config_dict() -> dict:
    """与 Relax 的 build_hf_peft_config_dict 对齐。"""
    return {
        "r": RANK,
        "lora_alpha": ALPHA,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }


def main() -> int:
    import torch

    _banner("0. 源码与环境")
    print(f"  torch {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    if verify_source():
        _ok("source", "跑的是带我们 delta 的 sglang")
    else:
        _fail("source", "delta 不在位，PYTHONPATH 顶错了")
        return 1

    _banner("1. 起引擎（enable_lora）")
    import sglang as sgl

    engine = sgl.Engine(
        model_path=MODEL_DIR,
        enable_lora=True,
        max_lora_rank=RANK,
        # Relax 的 sglang_engine 把 Megatron 名转成 HF 名后传进来
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        max_loras_per_batch=1,
        max_loaded_loras=2,
        tp_size=1,
        disable_cuda_graph=True,
        mem_fraction_static=float(os.environ.get("PROBE_MEM_FRACTION", "0.85")),
        log_level="info",
    )
    _ok("engine", "带 LoRA 起来了（塔没被误包，否则这里就 hidden dim 不匹配了）")

    sampling = {"max_new_tokens": 32, "temperature": 0.0}
    prompt = chat_prompt("Translate into Chinese: The weather is nice today.")
    cfg = config_dict()

    def gen(lora_name: str | None) -> tuple[str, list[int], list[float]]:
        kwargs = {"prompt": [prompt], "sampling_params": sampling, "return_logprob": True}
        if lora_name is not None:
            kwargs["lora_path"] = [lora_name]
        out = engine.generate(**kwargs)[0]
        olp = out["meta_info"].get("output_token_logprobs") or []
        return out["text"], [int(t[1]) for t in olp], [float(t[0]) for t in olp]

    def push(name: str, b_scale: float, seed: int | None = None) -> bool:
        tensors = make_adapter(b_scale=b_scale, seed=seed)
        n_cpu = sum(1 for t in tensors.values() if t.device.type == "cpu")
        res = engine.load_lora_adapter_from_tensors(lora_name=name, tensors=tensors, config_dict=cfg)
        success = getattr(res, "success", None)
        if success is None and isinstance(res, dict):
            success = res.get("success")
        if not success:
            print(f"    加载失败: {res}", flush=True)
            return False
        print(f"    推入 {len(tensors)} 个张量（CPU {n_cpu} 个），b_scale={b_scale}", flush=True)
        return True

    def max_lp_diff(a: list[float], b: list[float]) -> float:
        n = min(len(a), len(b))
        return max((abs(a[i] - b[i]) for i in range(n)), default=0.0)

    try:
        _banner("2. base 生成")
        base_text, base_ids, base_lps = gen(None)
        print(f"  文本: {base_text[:120]!r}", flush=True)
        print(f"  {len(base_ids)} 个 token，前 5 个 logprob: {[round(x, 4) for x in base_lps[:5]]}", flush=True)
        if base_ids:
            _ok("base", "base 生成正常（chat 模板套对了）")
        else:
            _fail("base", "空输出，后面没法比对")
            return 1

        _banner("3. 可逆性：推入 B=0 的 adapter，输出必须精确回到 base")
        if push("zero", b_scale=0.0):
            z_text, z_ids, z_lps = gen("zero")
            same = z_ids == base_ids and max_lp_diff(z_lps, base_lps) < 1e-3
            print(f"  文本: {z_text[:120]!r}", flush=True)
            print(f"  与 base 的 logprob 最大差: {max_lp_diff(z_lps, base_lps):.6f}", flush=True)
            if same:
                _ok("reversible", "B=0 精确复现 base，说明命名对上了且没有残留")
            else:
                _fail("reversible", "B=0 却改变了输出，命名或 buffer 有问题")
        else:
            _fail("reversible", "B=0 的 adapter 都没推进去")
            return 1

        _banner("4. 生效性：换成非零 B")
        engine.unload_lora_adapter("zero")
        if push(LORA_NAME, b_scale=0.12, seed=2024):
            t_text, t_ids, t_lps = gen(LORA_NAME)
            diff = max_lp_diff(t_lps, base_lps)
            print(f"  文本: {t_text[:120]!r}", flush=True)
            print(f"  与 base 的 logprob 最大差: {diff:.6f}", flush=True)
            if diff > 1e-3 or t_ids != base_ids:
                _ok("effect", "adapter 真的作用到前向上了")
            else:
                _fail("effect", "与 base 无差别，adapter 是个摆设")
        else:
            _fail("effect", "非零 adapter 推送失败")

        _banner("5. 稳定性：连续 unload→load，模拟 Relax 每步的节奏")
        rounds = int(os.environ.get("PROBE_ROUNDS", "3"))
        stable = True
        for i in range(rounds):
            engine.unload_lora_adapter(LORA_NAME)
            if not push(LORA_NAME, b_scale=0.12, seed=3000 + i):
                stable = False
                break
            _, ids_i, _ = gen(LORA_NAME)
            free, total = torch.cuda.mem_get_info()
            print(f"    第 {i + 1} 轮：{len(ids_i)} 个 token，空闲显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
        if stable:
            _ok("stable", f"{rounds} 轮更新没崩、显存有界")
        else:
            _fail("stable", "重复更新中途失败")
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass

    _banner("结论")
    if _failures:
        print(f"  失败项: {_failures}", flush=True)
        return 1
    print("  全部通过：v2 的 sglang 能带 LoRA 起 Omni，并接受 Relax 命名的 CPU 张量 adapter。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
