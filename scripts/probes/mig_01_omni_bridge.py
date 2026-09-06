"""Phase 0.1b —— 注册 Qwen3-Omni bridge 后重验 AutoBridge 识别（容器内运行）。

背景：mig_00_env.py 的第 4 个判据 [omni] 失败，报
    Model architecture 'Qwen3OmniMoeForConditionalGeneration' is not yet supported
但这不是真正的阻塞。Qwen3-Omni 的 bridge 不在 megatron-bridge 的内置支持列表里，
而是由 Relax 通过 @MegatronModelBridge.register_bridge 注册的：

    relax/models/__init__.py
        try:  from megatron.bridge.models.qwen_omni import Qwen3OmniMoEBridge
        except ImportError/AttributeError:
              from relax.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge

mig_00_env.py 沿用的是 2026-07 的验证逻辑（当时要验证的是"上游 bridge 是否原生支持
omni"），只 import 了 megatron.bridge，从未 import relax，所以注册从没发生。

本脚本验证注册后的真实能力：
  1. [bridge-builtin] megatron.bridge.models.qwen_omni 是否存在（钉的 bridge commit 自带？）
  2. [relax-import]   relax 能否 import，以及来源路径
  3. [register]       import relax.models 后 Qwen3-Omni 是否进入注册表
  4. [omni]           AutoBridge.from_hf_pretrained 能否识别 Qwen3-Omni
  5. [provider]       to_megatron_provider 能否构建 provider（不加载权重）

用法：python mig_01_omni_bridge.py   （默认读 /models/qwen3-omni，可用 OMNI_CKPT 覆盖）
"""

from __future__ import annotations

import os
import sys
import traceback

OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")
TARGET_ARCH = "Qwen3OmniMoeForConditionalGeneration"

_hard_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"[PASS][{tag}] {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"[FAIL][{tag}] {msg}", flush=True)
    _hard_failures.append(tag)


def _warn(tag: str, msg: str) -> None:
    print(f"[WARN][{tag}] {msg}", flush=True)


def _registry_snapshot() -> set[str]:
    """尽量把已注册的 HF 架构名捞出来。

    megatron-bridge 不同版本的注册表字段名可能不同，所以按候选名逐个试，
    并且失败不抛异常——这只是诊断信息，不作为判据。
    """
    try:
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    except Exception:  # noqa: BLE001
        return set()

    for attr in ("_BRIDGE_REGISTRY", "_registry", "BRIDGE_REGISTRY", "_bridges"):
        reg = getattr(MegatronModelBridge, attr, None)
        if isinstance(reg, dict) and reg:
            names = set()
            for k in reg:
                names.add(getattr(k, "__name__", str(k)))
            return names
    return set()


def check_bridge_builtin() -> None:
    """钉的 megatron-bridge commit 是否自带 qwen_omni（决定走 try 还是 except 分支）。"""
    tag = "bridge-builtin"
    try:
        from megatron.bridge.models.qwen_omni import Qwen3OmniMoEBridge  # noqa: F401

        _ok(tag, "megatron.bridge.models.qwen_omni 存在，Relax 会走 try 分支直接复用上游实现")
    except (ImportError, AttributeError) as e:
        _warn(tag, f"megatron.bridge.models.qwen_omni 不可用（{e!r}）；Relax 将回退到自带 bridge，属预期路径")


def check_relax_import() -> bool:
    """relax 是否可 import。镜像默认可能不含 relax 源码（README 让你自己 pip install -e .）。"""
    tag = "relax-import"
    try:
        import relax

        _ok(tag, f"relax 可导入 @ {getattr(relax, '__file__', '?')} (version={getattr(relax, '__version__', '?')})")
        return True
    except Exception:  # noqa: BLE001
        _fail(tag, "relax 无法导入（镜像未安装 relax 源码？）：\n" + traceback.format_exc())
        return False


def check_register() -> None:
    """import relax.models 触发 @register_bridge。"""
    tag = "register"
    before = _registry_snapshot()
    print(f"  注册表（import relax.models 之前）条目数: {len(before)}", flush=True)
    try:
        import relax.models  # noqa: F401
    except Exception:  # noqa: BLE001
        _fail(tag, "import relax.models 失败：\n" + traceback.format_exc())
        return

    after = _registry_snapshot()
    added = sorted(after - before)
    print(f"  注册表（之后）条目数: {len(after)}；新增: {added}", flush=True)
    if any("Omni" in n for n in after) or not after:
        # after 为空表示没捞到注册表字段（诊断手段失效），不据此判失败，交给 [omni] 实测。
        _ok(tag, "import relax.models 成功（是否真的注册以 [omni] 实测为准）")
    else:
        _warn(tag, "import relax.models 成功，但注册表里没看到 Omni 相关条目，以 [omni] 实测为准")


def check_omni() -> None:
    """注册后 AutoBridge 能否识别 Qwen3-Omni。"""
    tag = "omni"
    if not os.path.exists(os.path.join(OMNI_CKPT, "config.json")):
        _fail(tag, f"未找到 {OMNI_CKPT}/config.json，无法验证（请确认权重卷已挂载）")
        return
    try:
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(OMNI_CKPT, trust_remote_code=True)
        _ok(tag, f"AutoBridge 识别成功: {type(bridge).__name__}")
        globals()["_BRIDGE"] = bridge
    except Exception:  # noqa: BLE001
        _fail(tag, f"AutoBridge 仍无法识别 {TARGET_ARCH}：\n" + traceback.format_exc())


def check_provider() -> None:
    """provider 能否构建（不加载权重）。这是后续挂 LoRA 的前置条件。"""
    tag = "provider"
    bridge = globals().get("_BRIDGE")
    if bridge is None:
        _warn(tag, "[omni] 未通过，跳过 provider 构建")
        return
    try:
        provider = bridge.to_megatron_provider(load_weights=False)
        _ok(tag, f"provider 构建成功: {type(provider).__name__}")
        for attr in ("num_layers", "hidden_size", "num_moe_experts", "position_embedding_type"):
            if hasattr(provider, attr):
                print(f"    provider.{attr} = {getattr(provider, attr)}", flush=True)
    except Exception:  # noqa: BLE001
        _fail(tag, "to_megatron_provider 失败：\n" + traceback.format_exc())


def main() -> None:
    print("========== Phase 0.1b：注册后重验 Qwen3-Omni ==========", flush=True)
    check_bridge_builtin()
    if check_relax_import():
        check_register()
    check_omni()
    check_provider()

    print("\n========== 结论 ==========", flush=True)
    if _hard_failures:
        print(f"  [GATE X] 硬失败: {sorted(set(_hard_failures))}", flush=True)
        sys.exit(1)
    print("  [GATE OK] Qwen3-Omni 在新版 Relax 镜像上可被 bridge 识别并构建 provider。", flush=True)


if __name__ == "__main__":
    main()
