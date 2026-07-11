"""Phase 0.1 —— 上游 Megatron-Bridge 0.5.0 环境/API 兼容闸门（容器内运行）。

由 modal_migrate.py 在 Modal 容器里调起。对应 MIGRATION_PLAN.md 的 Phase 0.1。

验证四件事，任一硬失败 -> 退出码 1（迁移在此止步、维持 direct 路线）：
  1. [compat]     megatron.core 与 megatron.bridge 能在同一环境共存
                  —— 在 slime 镜像自带 megatron.core 之上叠加/替换成上游 bridge 0.5.0，
                     最容易在这里炸（API/版本不匹配）。这是"能不能迁"的第一道闸门。
  2. [export-api] AutoBridge 具备 adapter 导出 API
                  （export_adapter_ckpt / save_hf_adapter / export_adapter_weights 至少其一）。
  3. [peft]       bridge PEFT 的 LoRA / CanonicalLoRA 可导入。
  4. [omni]       AutoBridge.from_hf_pretrained 能识别本地 Qwen3-Omni checkpoint。

用法：python mig_00_env.py   （默认读 /models/qwen3-omni，可用 OMNI_CKPT 覆盖）
"""

from __future__ import annotations

import os
import sys
import traceback

OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

_hard_failures: list[str] = []
_warnings: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _hard_failures.append(tag)


def _warn(tag: str, msg: str) -> None:
    print(f"[WARN][{tag}] {msg}", flush=True)
    _warnings.append(tag)


def check_versions() -> None:
    print("========== 版本信息 ==========", flush=True)
    for mod in (
        "torch",
        "megatron.core",
        "megatron.bridge",
        "transformer_engine",
        "transformers",
        "peft",
    ):
        try:
            m = __import__(mod, fromlist=["__version__"])
            print(f"  {mod}: {getattr(m, '__version__', '?')} @ {getattr(m, '__file__', '?')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {mod}: <import 失败> {e!r}", flush=True)


def check_coexist() -> None:
    """闸门 1：megatron.core 与 megatron.bridge 共存。"""
    tag = "compat"
    try:
        import megatron.core as mcore  # noqa: F401
        import megatron.bridge as mb  # noqa: F401

        _ok(tag, f"megatron.core 与 megatron.bridge 共存 OK (bridge @ {getattr(mb, '__file__', '?')})")
    except Exception:  # noqa: BLE001
        _fail(tag, "megatron.core / megatron.bridge 无法共存：\n" + traceback.format_exc())


def check_export_api() -> None:
    """闸门 2：AutoBridge 具备 adapter 导出 API（PR #2574 引入，0.5.0 应有）。"""
    tag = "export-api"
    try:
        from megatron.bridge import AutoBridge
    except Exception:  # noqa: BLE001
        _fail(tag, "无法 import AutoBridge：\n" + traceback.format_exc())
        return

    have = {
        name: hasattr(AutoBridge, name)
        for name in ("export_adapter_ckpt", "save_hf_adapter", "export_adapter_weights")
    }
    print(f"  AutoBridge adapter API: {have}", flush=True)
    if any(have.values()):
        _ok(tag, f"具备 adapter 导出 API: {[k for k, v in have.items() if v]}")
    else:
        _fail(tag, "AutoBridge 没有任何 adapter 导出 API（0.5.0 应具备；检查是否装成了旧 fork）")


def check_peft() -> None:
    """闸门 3：bridge PEFT 的 LoRA 可导入。"""
    tag = "peft"
    try:
        from megatron.bridge.peft.lora import LoRA  # noqa: F401

        try:
            from megatron.bridge.peft.canonical_lora import CanonicalLoRA  # noqa: F401

            extra = " + CanonicalLoRA"
        except Exception:  # noqa: BLE001
            extra = "（无 CanonicalLoRA，仅 LoRA）"
            _warn(tag, "CanonicalLoRA 不可用（不阻断，标准 LoRA 够用）")
        _ok(tag, f"bridge PEFT LoRA 可导入{extra}")
    except Exception:  # noqa: BLE001
        _fail(tag, "无法 import bridge PEFT LoRA：\n" + traceback.format_exc())


def check_omni() -> None:
    """闸门 4：AutoBridge 能识别本地 Qwen3-Omni（未挂载权重时降级为 WARN 跳过）。"""
    tag = "omni"
    if not os.path.exists(os.path.join(OMNI_CKPT, "config.json")):
        _warn(tag, f"未找到 {OMNI_CKPT}/config.json，跳过 omni 识别（挂载权重卷后再测）")
        return
    try:
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(OMNI_CKPT, trust_remote_code=True)
        name = type(bridge).__name__

        prov = ""
        try:
            p = bridge.to_megatron_provider(load_weights=False)
            prov = f" -> provider {type(p).__name__}"
        except Exception as e:  # noqa: BLE001
            prov = f"（to_megatron_provider 失败: {e!r}）"
            _warn(tag, "from_hf_pretrained OK 但 to_megatron_provider 失败，见上")

        if "omni" in name.lower() or "omni" in prov.lower():
            _ok(tag, f"识别 Qwen3-Omni: AutoBridge={name}{prov}")
        else:
            _warn(tag, f"AutoBridge 构建成功但类名未含 omni: {name}{prov}（需人工确认是否走 omni 路径）")
    except Exception:  # noqa: BLE001
        _fail(tag, "from_hf_pretrained 识别 Qwen3-Omni 失败：\n" + traceback.format_exc())


def main() -> None:
    check_versions()
    print("\n========== Phase 0.1 检查 ==========", flush=True)
    check_coexist()
    check_export_api()
    check_peft()
    check_omni()

    print("\n========== 结论 ==========", flush=True)
    if _warnings:
        print(f"  警告: {sorted(set(_warnings))}", flush=True)
    if _hard_failures:
        print(f"  [GATE X] 硬失败: {sorted(set(_hard_failures))} —— 迁移在 Phase 0.1 止步。", flush=True)
        sys.exit(1)
    print("  [GATE OK] Phase 0.1 全部通过 —— 可进入 Phase 0.2（omni thinker 挂 LoRA）。", flush=True)


if __name__ == "__main__":
    main()
