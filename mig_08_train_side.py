"""探针七 —— 训练侧冒烟：Bridge 建 Omni + LoRA 注入 + adapter 导出（容器内运行，1 卡）。

到目前为止验的全是推理侧（探针四、五）和导出命名的静态预测（探针二）。训练侧
在 v2 上一次都没跑过，未知都堆在这边。这个探针只回答三个问题，完全不碰 rollout、
数据集、优化器：

  1. Megatron-Bridge 能不能把 Qwen3-Omni 建起来（走 Relax 注册的 Qwen3OmniMoEBridge）
  2. LoRA 能不能按 linear_qkv / linear_proj 注进去，且只注到语言模型上
  3. export_adapter_weights 导出的命名和形状，是不是探针二预测的那套

省钱的关键是缩层：provider.num_layers = 2，随机初始化，不加载 30B 的真实权重。
这不是我们发明的 hack —— Relax 的 model_provider.py 把 num_layers 列进了
bridge_keys，注释写明「Allow CLI to override layer count for layer-reduced
training」，是官方支持的路子。这样一张 A10G 就够，而不是 A100。

跑法：modal run modal_probe_train_side.py
"""

from __future__ import annotations

import os
import traceback
from types import SimpleNamespace


OMNI_CKPT = os.environ.get("OMNI_CKPT", "/models/qwen3-omni")

# 与 v1 冒烟脚本一致的 LoRA 配置。target modules 用 Megatron 侧的名字，
# Megatron-Bridge 的 matcher 走的就是这套名字。
LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj"]
LORA_DROPOUT = 0.0

NUM_LAYERS = 2

_failures: list[str] = []


def _ok(tag: str, msg: str) -> None:
    print(f"  [PASS] {tag}: {msg}", flush=True)


def _fail(tag: str, msg: str) -> None:
    print(f"  [FAIL] {tag}: {msg}", flush=True)
    _failures.append(tag)


def _banner(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


def init_distributed() -> None:
    """单进程的 dist + Megatron 并行状态。provide() 里的模块在构造时就要读
    parallel_state，所以必须在建模型之前初始化。"""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    print("  dist + parallel_state 就绪（tp=1 pp=1 ep=1）", flush=True)


def build_provider():
    """走 Relax 的路径拿 provider：AutoBridge -> to_megatron_provider(load_weights=False)。

    与 relax/backends/megatron/model_provider.py 的 bridge 分支同一套动作，只是
    我们手工设那几个 override，而不是从完整的 args namespace 里搬。
    """
    import torch

    # 注册 Qwen3OmniMoEBridge，否则 AutoBridge 认不出 Qwen3OmniMoeForConditionalGeneration
    import relax.models.qwen_omni.qwen3_omni_bridge  # noqa: F401
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(OMNI_CKPT, trust_remote_code=True)
    print(f"  bridge = {type(bridge).__name__}", flush=True)

    provider = bridge.to_megatron_provider(load_weights=False)
    print(f"  provider = {type(provider).__name__}  原始层数 = {provider.num_layers}", flush=True)

    provider.num_layers = NUM_LAYERS
    # moe_layer_freq 若是逐层列表，长度必须跟着缩，否则 finalize 时对不上
    freq = getattr(provider, "moe_layer_freq", None)
    if isinstance(freq, list):
        provider.moe_layer_freq = freq[:NUM_LAYERS]
        print(f"  moe_layer_freq 列表截到 {len(provider.moe_layer_freq)} 项", flush=True)

    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = False
    provider.fp16 = False
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16

    provider.finalize()
    print(f"  finalize 完成，缩后层数 = {provider.num_layers}", flush=True)
    return bridge, provider


def check_lora_scope(model) -> None:
    """LoRA 只能落在语言模型上。塔里带 adapter 就说明 target 匹配漏了。

    这是探针一的训练侧真实版：探针一是在 meta device 上拿 HF 编码器和
    ModuleMatcher 对了一遍，这里是 Relax 真建出来的 Megatron 模型。
    """
    adapter_params = [n for n, _ in model.named_parameters() if "adapter" in n]
    if not adapter_params:
        _fail("inject", "一个 adapter 参数都没有，LoRA 没注进去")
        return

    tower_hits = [n for n in adapter_params if n.startswith(("audio_model", "vision_model"))]
    if tower_hits:
        _fail("scope", f"塔里出现了 {len(tower_hits)} 个 adapter 参数，例如 {tower_hits[:3]}")
    else:
        _ok("scope", f"{len(adapter_params)} 个 adapter 参数全在语言模型上，塔干净")

    print("  前 6 个 adapter 参数：", flush=True)
    for n in adapter_params[:6]:
        print(f"    {n}", flush=True)

    per_layer = len(adapter_params) / max(NUM_LAYERS, 1)
    print(f"  每层 {per_layer:.1f} 个（linear_qkv + linear_proj，各有 A/B，预期 4）", flush=True)


def check_frozen(model) -> None:
    """PEFT 应该把底模全冻上，只留 adapter 可训。"""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    non_adapter = [n for n in trainable if "adapter" not in n]
    if non_adapter:
        _fail("freeze", f"{len(non_adapter)} 个非 adapter 参数还可训，例如 {non_adapter[:3]}")
    else:
        _ok("freeze", f"只有 {len(trainable)} 个 adapter 参数可训，底模已冻")


def check_export(bridge, model) -> None:
    """导出 adapter，核对命名与形状。

    对照探针二的预测：thinker.<...>.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight。
    Megatron 侧是 fused 的 linear_qkv，Bridge 的 QKVMapping 负责拆成 q/k/v —— 这
    一步是 v1 手写 de-interleave 的替代品，正是要确认的地方。
    """
    from relax.utils import megatron_bridge_utils
    from relax.utils.megatron_peft_utils import convert_megatron_to_hf_target_modules

    with megatron_bridge_utils.patch_megatron_model([model]):
        items = list(bridge.export_adapter_weights([model], cpu=True, show_progress=False))

    if not items:
        _fail("export", "导出为空")
        return

    names = [it.param_name for it in items]
    print(f"  导出 {len(names)} 个张量，全部列出：", flush=True)
    shapes = {}
    for it in items:
        shapes[it.param_name] = tuple(it.weight.shape)
        print(f"    {it.param_name}  {tuple(it.weight.shape)}", flush=True)

    bad_prefix = [n for n in names if not n.startswith("thinker.")]
    if bad_prefix:
        _fail("export_prefix", f"{len(bad_prefix)} 个名字不以 thinker. 开头，例如 {bad_prefix[:3]}")
    else:
        _ok("export_prefix", "全部以 thinker. 开头（SGLang 侧的模块名就是这个前缀）")

    tower = [n for n in names if "audio_tower" in n or "visual" in n]
    if tower:
        _fail("export_scope", f"导出里混进了塔的权重：{tower[:3]}")
    else:
        _ok("export_scope", "导出里没有塔的权重")

    leaves = sorted({n.split(".")[-3] for n in names if n.endswith((".lora_A.weight", ".lora_B.weight"))})
    print(f"  叶子模块：{leaves}", flush=True)
    expected_leaves = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if set(leaves) == expected_leaves:
        _ok("export_split", "fused linear_qkv 已被 Bridge 拆成 q/k/v_proj —— v1 手写的 de-interleave 可以退休")
    else:
        _fail("export_split", f"叶子模块与预期不符，预期 {sorted(expected_leaves)}")

    ab = [n for n in names if ".lora_A." in n or ".lora_B." in n]
    if len(ab) != len(names):
        _fail("export_ab", f"有 {len(names) - len(ab)} 个名字既不是 lora_A 也不是 lora_B")
    else:
        _ok("export_ab", "全部是 lora_A / lora_B")

    for n, s in shapes.items():
        if ".lora_A." in n and s[0] != LORA_RANK:
            _fail("export_shape", f"{n} 的 lora_A 首维应为 rank={LORA_RANK}，实际 {s}")
            break
        if ".lora_B." in n and s[1] != LORA_RANK:
            _fail("export_shape", f"{n} 的 lora_B 次维应为 rank={LORA_RANK}，实际 {s}")
            break
    else:
        _ok("export_shape", f"A 为 [r={LORA_RANK}, in]、B 为 [out, r={LORA_RANK}]")

    hf_targets = convert_megatron_to_hf_target_modules(LORA_TARGET_MODULES)
    print(f"\n  adapter_config.json 里会写的 target_modules: {hf_targets}", flush=True)
    covered = {n.split(".")[-3] for n in names}
    missing = covered - set(hf_targets)
    if missing:
        _fail("config_targets", f"导出里有 {sorted(missing)}，但 target_modules 没写进去")
    else:
        _ok("config_targets", "导出的叶子模块都被 target_modules 覆盖")


def main() -> int:
    import torch

    _banner("0. 环境")
    print(f"  torch {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    import megatron.bridge
    import relax

    print(f"  relax 来自 {os.path.dirname(relax.__file__)}", flush=True)
    print(f"  megatron.bridge 来自 {os.path.dirname(megatron.bridge.__file__)}", flush=True)

    _banner("1. 初始化并行状态")
    init_distributed()

    _banner("2. Bridge 建 provider（缩到 2 层，不加载权重）")
    try:
        bridge, provider = build_provider()
    except Exception as exc:  # noqa: BLE001
        _fail("provider", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("3. 建模型")
    try:
        model = provider.provide(pre_process=True, post_process=True)
        n_params = sum(p.numel() for p in model.parameters())
        free, total = torch.cuda.mem_get_info()
        print(f"  参数量 {n_params / 1e9:.2f} B，空闲显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
        _ok("build", f"{type(model).__name__} 建起来了")
    except Exception as exc:  # noqa: BLE001
        _fail("build", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("4. 注入 LoRA")
    try:
        from relax.utils.megatron_peft_utils import build_lora_peft

        args = SimpleNamespace(
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
        )
        peft = build_lora_peft(args)
        print(f"  peft = {type(peft).__name__}  target_modules={LORA_TARGET_MODULES}", flush=True)
        model = peft(model, training=True)
        _ok("peft", "LoRA 已应用")
    except Exception as exc:  # noqa: BLE001
        _fail("peft", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _banner("5. 注入范围与冻结")
    check_lora_scope(model)
    check_frozen(model)

    _banner("6. 导出 adapter")
    try:
        check_export(bridge, model)
    except Exception as exc:  # noqa: BLE001
        _fail("export", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    _banner("结论")
    if _failures:
        print(f"  失败项: {sorted(set(_failures))}", flush=True)
        return 1
    print("  训练侧三件事都成立：Bridge 能建 Omni、LoRA 只落在语言模型上、导出命名与 SGLang 对得上。", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
