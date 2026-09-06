"""探针六 —— CPU 张量走 MultiprocessingSerializer 的越界问题（纯 CPU，几分钟）。

给第三个 sglang PR 攒实证。要回答三件事：

  1. 一个 CPU 张量经 torch 的 reduce_tensor 会变成几元组？index 6 到底是什么？
     （上游 _reduce_tensor_modified 无条件改写 index 6，靠的是"签名多年没变"的假设，
      但那个假设只对 CUDA 张量成立）
  2. 装了 reducer 之后序列化 CPU 张量，上游版本会怎么炸？
  3. 我们加的长度守卫能不能让它正常通过？

因果链要说清楚：v1 在 sglang 0.5.9 上推 CPU 张量没事，是因为当时 LoRA 那条路
压根没调 monkey_patch_torch_reductions。是我们给 tp_worker 补上 reducer 安装
之后，这个越界才暴露出来 —— 所以那两处改动是绑在一起的，不能只提其中一个。

verl 踩过同一个坑（bug #4065，CPU tensor 序列化）。
"""

from __future__ import annotations

import sys
import traceback


_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"  [PASS] {tag}: {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"  [FAIL] {tag}: {msg}", flush=True)
    _failures.append(tag)


def _banner(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def get_serializer():
    try:
        from sglang.srt.utils import MultiprocessingSerializer

        return MultiprocessingSerializer
    except ImportError:
        from sglang.srt.utils.common import MultiprocessingSerializer

        return MultiprocessingSerializer


def main() -> int:
    import os

    import torch
    from torch.multiprocessing import reductions

    import sglang
    from sglang.srt.utils import patch_torch as pt

    _banner("0. 环境")
    print(f"  sglang {sglang.__version__} 来自 {os.path.dirname(sglang.__file__)}", flush=True)
    print(f"  torch {torch.__version__}  CUDA 可用: {torch.cuda.is_available()}", flush=True)
    guarded_src = "if len(output_args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:" in open(
        pt.__file__, encoding="utf-8"
    ).read()
    print(f"  patch_torch 带长度守卫: {guarded_src}", flush=True)
    if not guarded_src:
        _fail("source", "跑的不是带守卫的那份 patch_torch")
        return 1

    _banner("1. CPU 张量 reduce 出来长什么样")
    tensor = torch.zeros(4, 4, dtype=torch.bfloat16)
    fn, args = reductions.reduce_tensor(tensor)
    print(f"  rebuild 函数: {fn.__name__}", flush=True)
    print(f"  参数元组长度: {len(args)}   （上游要改写的是 index {pt._REDUCE_TENSOR_ARG_DEVICE_INDEX}）", flush=True)
    for i, a in enumerate(args):
        kind = type(a).__name__
        shown = a if isinstance(a, (int, bool, tuple, type(None))) else f"<{kind}>"
        mark = "  <-- 上游要改写这一项" if i == pt._REDUCE_TENSOR_ARG_DEVICE_INDEX else ""
        print(f"    [{i}] {kind}: {shown}{mark}", flush=True)

    if len(args) > pt._REDUCE_TENSOR_ARG_DEVICE_INDEX:
        print("  注意：长度足够，不会 IndexError，但 index 6 并不是 device —— 会被写坏。", flush=True)
    else:
        print("  长度不足 7，上游那行会直接 IndexError。", flush=True)

    _banner("2. 装上 reducer 后序列化 CPU 张量（我们的守卫版）")
    pt.monkey_patch_torch_reductions()
    ser = get_serializer()
    payload = {"thinker.model.layers.0.self_attn.q_proj.lora_A.weight": tensor}
    try:
        blob = ser.serialize(payload, output_str=True)
        _ok("guarded", f"序列化通过，产物 {len(blob)} 字符")
    except Exception as exc:  # noqa: BLE001
        _fail("guarded", f"带守卫竟然也炸了: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("3. 换成上游那版（无守卫），同样的输入")

    def _reduce_tensor_upstream(*a, **kw):
        """上游 sglang 的原样实现：无条件改写 index 6。"""
        output_fn, output_args = reductions._reduce_tensor_original(*a, **kw)
        output_args = pt._modify_tuple(output_args, pt._REDUCE_TENSOR_ARG_DEVICE_INDEX, pt._device_to_uuid)
        return output_fn, output_args

    reductions.reduce_tensor = _reduce_tensor_upstream
    reductions.init_reductions()
    try:
        ser.serialize(payload, output_str=True)
        _fail("upstream", "上游版本居然没报错 —— 结论要重新审")
    except Exception as exc:  # noqa: BLE001
        print(f"  抛出: {type(exc).__name__}: {exc}", flush=True)
        _ok("upstream", "上游版本在 CPU 张量上确实炸了")
    finally:
        reductions.reduce_tensor = pt._reduce_tensor_modified
        reductions.init_reductions()

    _banner("4. 回到守卫版，确认可恢复")
    try:
        ser.serialize(payload, output_str=True)
        _ok("restore", "换回守卫版后又能正常序列化")
    except Exception as exc:  # noqa: BLE001
        _fail("restore", f"{type(exc).__name__}: {exc}")

    _banner("结论")
    if _failures:
        print(f"  失败项: {_failures}", flush=True)
        return 1
    print("  CPU 张量在装了 reducer 的进程里，上游版本会炸，守卫版本正常。", flush=True)
    print("  这就是 tp_worker 补装 reducer 之后必须同时带上 patch_torch 守卫的原因。", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
