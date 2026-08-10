"""探针一 —— LoRA 会挂到 Qwen3-Omni 的哪些模块上（容器内运行）。

要回答的问题：
    用官方 Relax 的 LoRA（Megatron-Bridge PEFT），adapter 会不会挂到
    audio / vision tower 上？

静态分析已经给出预期结论，本脚本是去证实它，不是重新探索：

  1. relax/models/qwen_omni/modeling_qwen3_omni/model.py 里，
     Qwen3OmniMoeModel 在 pre_process 时把 audio_model / vision_model 建成
     **transformers 的 HF 实现**（Qwen3OmniMoeAudioEncoder / VisionEncoder），
     只有 language_model 是 Megatron 的 GPT。所以两个塔确实在 Megatron 模型树里。

  2. Megatron-Bridge 的 ModuleMatcher 在 target_modules 非空时，只按
     「叶子名精确相等」或「full_name 通配符匹配」来判定，不看模块类型。
     于是塔会不会被挂上，完全取决于 HF 编码器里的线性层叫什么名字。

  3. 查 transformers 的实现：
       audio tower  —— q_proj / k_proj / v_proj / out_proj / fc1 / fc2 / proj1 / proj2
       vision attn  —— qkv / proj
       vision MLP   —— linear_fc1 / linear_fc2   <-- Megatron 风格的名字！
       merger       —— linear_fc1 / linear_fc2   <-- 同上
     所以预期是：
       * 只挂 linear_qkv + linear_proj（v1 用的 Q/K/V/O）  -> 两个塔都不会被碰
       * 一旦把 linear_fc1 / linear_fc2 加进去            -> 视觉塔和 merger 会被静默挂上

预期结论若成立，意味着不需要改 Relax 的注入逻辑，只要把 target modules 限制在
注意力投影上即可；反之则需要用通配符或 exclude_modules 限定范围。

便宜的做法：
    塔用 meta device 构建（不分配显存、不加载权重、层数压到 1），只走匹配器。
    整个脚本不建 Megatron 模型、不初始化分布式。

用法：python mig_02_lora_scope.py   （默认读 /models/qwen3-omni，可用 OMNI_CKPT 覆盖）
"""

from __future__ import annotations

import inspect
import os
import sys
import traceback

OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

# 要对比的几组 target modules。key 是说明，value 是传给 --lora-target-modules 的值。
TARGET_SETS: dict[str, list[str]] = {
    "A. v1 与官方默认：只挂注意力 Q/K/V/O": ["linear_qkv", "linear_proj"],
    "B. 追加 MLP（危险组合）": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
}

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _failures.append(tag)


def _warn(tag: str, msg: str) -> None:
    print(f"[WARN][{tag}] {msg}", flush=True)


def step_versions() -> None:
    """确认镜像里的 transformers 与我们查过的实现是同一份。

    静态结论建立在 transformers 的 Qwen3OmniMoe* 实现上，版本不同名字可能变，
    所以先把版本号打出来存档。
    """
    tag = "versions"
    try:
        import torch
        import transformers

        print(f"    torch        = {torch.__version__}", flush=True)
        print(f"    transformers = {transformers.__version__}", flush=True)
        try:
            import megatron.bridge as mb

            print(f"    megatron.bridge = {getattr(mb, '__version__', '?')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    megatron.bridge 版本读取失败: {e!r}", flush=True)
        _ok(tag, "版本已记录")
    except Exception:  # noqa: BLE001
        _fail(tag, "基础包导入失败：\n" + traceback.format_exc())


def step_walk_source() -> None:
    """确认 PEFT 的遍历真的会下探到 HF 子模块。

    这是整个判断的前提：如果 PEFT.__call__ 只走 Megatron 自己的模块，
    那两个 HF 塔压根不在遍历范围内，名字撞不撞都无所谓。
    """
    tag = "walk"
    try:
        from megatron.bridge.peft.base import PEFT

        src = inspect.getsource(PEFT.__call__)
        print("---- PEFT.__call__ 源码 ----", flush=True)
        print(src, flush=True)
        try:
            walk_fn = getattr(PEFT, "walk", None)
            if walk_fn is not None:
                print("---- PEFT.walk 源码 ----", flush=True)
                print(inspect.getsource(walk_fn), flush=True)
        except Exception:  # noqa: BLE001
            pass
        _ok(tag, "已打印遍历实现，请确认它是对 named_children / named_modules 的通用递归")
    except Exception:  # noqa: BLE001
        _fail(tag, "无法读取 PEFT 遍历实现：\n" + traceback.format_exc())


def _shrink(cfg, candidates: list[str], value: int = 1) -> None:
    """把层数类字段压到 1，纯粹为了建得快、占得少。"""
    for attr in candidates:
        if hasattr(cfg, attr):
            old = getattr(cfg, attr)
            setattr(cfg, attr, value)
            print(f"    压缩 {type(cfg).__name__}.{attr}: {old} -> {value}", flush=True)


def build_towers():
    """用真实 config 在 meta device 上建两个塔，只为拿到模块名。

    meta device 不分配任何真实存储，所以这一步几乎不花钱；
    模块名与真实构建完全一致。
    """
    tag = "towers"
    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
            Qwen3OmniMoeVisionEncoder,
        )

        cfg = AutoConfig.from_pretrained(OMNI_CKPT, trust_remote_code=True)
        thinker = getattr(cfg, "thinker_config", cfg)
        audio_cfg = getattr(thinker, "audio_config", None)
        vision_cfg = getattr(thinker, "vision_config", None)
        if audio_cfg is None or vision_cfg is None:
            _fail(tag, f"config 里找不到 audio_config / vision_config（thinker={type(thinker).__name__}）")
            return {}

        _shrink(audio_cfg, ["encoder_layers", "num_hidden_layers", "n_layer"])
        _shrink(vision_cfg, ["depth", "num_hidden_layers", "n_layer"])

        towers = {}
        with torch.device("meta"):
            towers["audio_model"] = Qwen3OmniMoeAudioEncoder._from_config(audio_cfg)
            towers["vision_model"] = Qwen3OmniMoeVisionEncoder._from_config(vision_cfg)
        _ok(tag, f"两个塔已在 meta device 上构建: {[type(t).__name__ for t in towers.values()]}")
        return towers
    except Exception:  # noqa: BLE001
        _fail(tag, "构建塔失败：\n" + traceback.format_exc())
        return {}


def step_linear_inventory(towers: dict) -> None:
    """列出两个塔里所有线性层的叶子名，作为撞名判断的事实依据。"""
    tag = "inventory"
    try:
        import torch.nn as nn

        for root_name, tower in towers.items():
            leaf_names: dict[str, int] = {}
            for name, mod in tower.named_modules():
                if isinstance(mod, nn.Linear):
                    leaf = name.rpartition(".")[2]
                    leaf_names[leaf] = leaf_names.get(leaf, 0) + 1
            print(f"    {root_name} 的 nn.Linear 叶子名:", flush=True)
            for leaf, n in sorted(leaf_names.items()):
                flag = "  <-- 与 Megatron 命名撞车" if leaf.startswith("linear_") else ""
                print(f"        {leaf:<24} x{n}{flag}", flush=True)
        _ok(tag, "清单已列出")
    except Exception:  # noqa: BLE001
        _fail(tag, "清点线性层失败：\n" + traceback.format_exc())


def step_matcher(towers: dict) -> None:
    """用真实的 ModuleMatcher 判定每组 target modules 会命中塔里的哪些模块。

    直接调 ModuleMatcher.match，走的就是 LoRA.transform 里那一行
    `if (ans := self.match(module, name, prefix)) is not None`，
    所以结论等价于真实注入，但不需要建 Megatron 模型。
    """
    tag = "matcher"
    try:
        from megatron.bridge.peft.module_matcher import ModuleMatcher

        for label, targets in TARGET_SETS.items():
            print(f"\n---- {label} ----", flush=True)
            print(f"    --lora-target-modules {' '.join(targets)}", flush=True)
            total_hits = 0
            for root_name, tower in towers.items():
                matcher = ModuleMatcher(target_modules=list(targets))
                matcher._init_target_match_state()
                hits: list[str] = []
                for name, mod in tower.named_modules():
                    if not name:
                        continue
                    prefix, _, leaf = name.rpartition(".")
                    full_prefix = f"{root_name}.{prefix}" if prefix else root_name
                    if matcher.match(mod, leaf, full_prefix) is not None:
                        hits.append(f"{full_prefix}.{leaf}  ({type(mod).__name__})")
                total_hits += len(hits)
                if hits:
                    print(f"    [{root_name}] 命中 {len(hits)} 个模块:", flush=True)
                    for h in hits[:20]:
                        print(f"        {h}", flush=True)
                    if len(hits) > 20:
                        print(f"        ... 另有 {len(hits) - 20} 个", flush=True)
                else:
                    print(f"    [{root_name}] 未命中任何模块", flush=True)
            verdict = "塔会被挂上 LoRA" if total_hits else "塔不会被碰"
            print(f"    => {verdict}（命中总数 {total_hits}）", flush=True)
        _ok(tag, "匹配判定完成")
    except Exception:  # noqa: BLE001
        _fail(tag, "匹配判定失败：\n" + traceback.format_exc())


def main() -> None:
    print("========== 探针一：LoRA 作用范围 ==========", flush=True)
    print("\n[1] 环境版本", flush=True)
    step_versions()

    print("\n[2] PEFT 遍历是否下探到 HF 子模块", flush=True)
    step_walk_source()

    print("\n[3] 构建 audio / vision 塔（meta device）", flush=True)
    towers = build_towers()

    if towers:
        print("\n[4] 塔内线性层命名清单", flush=True)
        step_linear_inventory(towers)

        print("\n[5] 各组 target modules 的命中情况", flush=True)
        step_matcher(towers)

    print("\n========== 结论 ==========", flush=True)
    if _failures:
        print(f"  [PROBE X] 失败环节: {sorted(set(_failures))}", flush=True)
        sys.exit(1)
    print("  [PROBE OK] 见上方 [5] 的命中情况：", flush=True)
    print("    A 组不命中 => 只挂注意力时无需改 Relax 注入逻辑。", flush=True)
    print("    B 组命中   => 加 MLP 前必须用通配符或 exclude_modules 限定范围。", flush=True)


if __name__ == "__main__":
    main()
