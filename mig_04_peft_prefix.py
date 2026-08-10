"""探针三 —— Relax 导出的 adapter 目录能不能被标准 PEFT 读回（容器内运行，纯 CPU）。

背景：
    relax/backends/megatron/checkpoint.py 里 _save_lora_to_checkpoint 的说法是，
    checkpoint 下的 lora_adapter/ 是「可移植的导出产物，供外部/推理使用，
    例如用 peft.PeftModel.from_pretrained 加载」，且明确不是续训来源。
    但 write_hf_peft_adapter 是把 export_adapter_weights 的 param_name 原样落盘的，
    而那批名字**不带** base_model.model. 前缀（上游只在
    convert_adapter_weights_to_peft_state 里才加）。

要回答：
    1. 标准 PEFT 自己写出来的 key 长什么样？（确认前缀确实存在，也顺便确认
       v1 用的 base_model.model.thinker.model.layers... 是符合 PEFT 规范的）
    2. 把前缀去掉之后，PeftModel.from_pretrained 还能不能把权重真正装进去？

怎么判定「真正装进去」：
    PEFT 初始化时 lora_B 恒为全零。所以先把 lora_B 全填成 1.0 再存；
    读回后如果 lora_B 仍是全零，就说明权重根本没落到位（而且是静默的）。

用法：python mig_04_peft_prefix.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _failures.append(tag)


def _build_base():
    """一个极小的因果语言模型，只为拿到真实的 PEFT 包装行为。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(cfg)


def _lora_b_absmax(peft_model) -> float:
    import torch

    vals = [
        p.detach().abs().max().item()
        for n, p in peft_model.named_parameters()
        if "lora_B" in n
    ]
    return max(vals) if vals else float("nan")


def main() -> None:
    print("========== 探针三：PEFT 前缀契约 ==========", flush=True)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file

    import peft as peft_mod
    import transformers

    print(f"    transformers = {transformers.__version__}", flush=True)
    print(f"    peft         = {peft_mod.__version__}", flush=True)
    print(f"    torch        = {torch.__version__}", flush=True)

    workdir = Path(tempfile.mkdtemp(prefix="peft-prefix-"))
    dir_std = workdir / "standard"
    dir_stripped = workdir / "stripped"

    # ---- 1. 让标准 PEFT 自己写一份，看它的 key 长什么样 ----
    tag = "peft-writes"
    try:
        model = get_peft_model(
            _build_base(),
            LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
        )
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.fill_(1.0)
        print(f"    存盘前 lora_B 绝对值最大 = {_lora_b_absmax(model)}", flush=True)
        model.save_pretrained(dir_std.as_posix())

        std_keys = sorted(load_file((dir_std / "adapter_model.safetensors").as_posix()).keys())
        print("\n    PEFT 自己写出的 key:", flush=True)
        for k in std_keys:
            print(f"        {k}", flush=True)

        prefixed = [k for k in std_keys if k.startswith("base_model.model.")]
        if len(prefixed) != len(std_keys):
            _fail(tag, "并非所有 key 都带 base_model.model. 前缀，前提假设不成立")
            return
        _ok(tag, f"{len(std_keys)} 个 key 全部带 base_model.model. 前缀")
    except Exception:  # noqa: BLE001
        _fail(tag, "让 PEFT 写盘失败：\n" + traceback.format_exc())
        return

    # ---- 2. 造一份「去掉前缀」的，模拟 Relax 的落盘方式 ----
    tag = "strip"
    try:
        shutil.copytree(dir_std, dir_stripped)
        state = load_file((dir_std / "adapter_model.safetensors").as_posix())
        stripped = {k[len("base_model.model.") :]: v for k, v in state.items()}
        save_file(stripped, (dir_stripped / "adapter_model.safetensors").as_posix())
        print("\n    去前缀后的 key（Relax 落盘的形态）:", flush=True)
        for k in sorted(stripped):
            print(f"        {k}", flush=True)
        _ok(tag, "已构造去前缀版本，adapter_config.json 与标准版完全一致")
    except Exception:  # noqa: BLE001
        _fail(tag, "构造去前缀版本失败：\n" + traceback.format_exc())
        return

    # ---- 3. 两份分别读回，看 lora_B 有没有真的落进去 ----
    results: dict[str, float] = {}
    for label, path in (("带前缀（标准 PEFT）", dir_std), ("不带前缀（Relax 形态）", dir_stripped)):
        tag = "load"
        try:
            loaded = PeftModel.from_pretrained(_build_base(), path.as_posix())
            absmax = _lora_b_absmax(loaded)
            results[label] = absmax
            verdict = "权重已装入" if absmax > 0 else "lora_B 仍是全零 —— 什么都没装进去"
            print(f"\n    {label}: lora_B 绝对值最大 = {absmax}  -> {verdict}", flush=True)
        except Exception:  # noqa: BLE001
            results[label] = float("nan")
            print(f"\n    {label}: 加载抛异常\n{traceback.format_exc()}", flush=True)

    # ---- 4. 判定 ----
    print("\n========== 结论 ==========", flush=True)
    std_val = results.get("带前缀（标准 PEFT）", 0.0)
    strip_val = results.get("不带前缀（Relax 形态）", 0.0)

    if not (std_val > 0):
        _fail("verdict", "连标准 PEFT 自己写的都没读回来，测试本身有问题，结论不可信")
    elif strip_val > 0:
        print("  [结论] PEFT 能容忍缺少 base_model.model. 前缀，Relax 的落盘方式没问题。", flush=True)
    else:
        print("  [结论] PEFT 不认缺前缀的 key：加载静默成功但 lora_B 全零，", flush=True)
        print("         也就是说 Relax 导出的 lora_adapter/ 用标准 PEFT 加载会得到一个", flush=True)
        print("         「什么都没学到」的模型。这是给 Relax 开 PR 的第二个点。", flush=True)
        print("         注意：不影响 SGLang（它按后缀和正则匹配），也不影响续训", flush=True)
        print("         （续训走原生 torch_dist checkpoint）。", flush=True)

    cfg_path = dir_std / "adapter_config.json"
    print(f"\n    参考：PEFT 写的 adapter_config.json = {json.loads(cfg_path.read_text())}", flush=True)

    if _failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
