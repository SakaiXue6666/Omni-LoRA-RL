# PR 1 —— sgl-project/sglang

分支：`SakaiXue6666:fix/lora-honor-should-apply-lora`（基于 main @ `f148eb6e6e`）

开 PR 链接（浏览器已登录即可）：
https://github.com/sgl-project/sglang/compare/main...SakaiXue6666:sglang:fix/lora-honor-should-apply-lora?expand=1

---

## Title

```
Honor should_apply_lora when wrapping LoRA target modules
```

---

## Body（直接粘贴）

## Motivation

`LoRAManager.init_lora_modules()` decides what to wrap by matching the last one or
two components of a module name against `--lora-target-modules`. For multimodal
models that is ambiguous: encoder towers name their projections the same way the
language model does. `VisionAttention` exposes `qkv_proj` and `proj`, so a target
as ordinary as `qkv_proj` selects tower modules the adapter carries no weights
for, and `get_layer_id()` files them into `self.lora_modules[<layer>]` next to the
language model's modules for that layer.

Models already declare the intended scope through `should_apply_lora`, and several
say so explicitly — `mllama4`: *"Skip vision model and multi_modal_projector for
LoRA"*, `gemma3_mm`: *"Skip vision tower and multi_modal_projector for LoRA"*,
`qwen2_vl`: *"skip visual tower"*, `ernie45_vl`: *"skip vision_model"*. The
comment inside `init_lora_modules()` also refers to the hook:

```python
# Handle embed_tokens and lm_head before the should_apply_lora gate,
# since VL models' should_apply_lora patterns only match language
# model layers and would incorrectly skip these.
```

But there is no call site anywhere in the tree. Thirteen model files define
`should_apply_lora` and none of them has any effect; the declared scope is not
enforced, and the special-casing of `embed_tokens` / `lm_head` guards against a
gate that never runs.

## Modifications

Restore the call between the special-cased modules and the suffix match, which is
exactly where the existing comment says it belongs:

```python
should_apply_lora = getattr(self.base_model, "should_apply_lora", None)
if callable(should_apply_lora) and not should_apply_lora(module_name):
    continue
```

Models that do not define the hook keep the plain suffix behavior, so nothing
changes for them.

Added `test/registered/unit/lora/test_should_apply_lora_gate.py`, which pins three
behaviors: a tower reusing the language model's names stays unwrapped, a model
without the hook keeps suffix matching, and a deny-all hook wraps nothing. The
tests build `LoRAManager` through `__new__`, so no memory pool, adapter download
or CUDA setup is involved. They pass with this change and fail without it.

One behavior change worth flagging: `interns2_mobius` declares
`should_apply_lora` as `module_name.startswith("model.layers.")`, so with the gate
active its `model.meta_mlp.*` modules are skipped before reaching the
`FusedMoE` branch that currently raises a descriptive `ValueError` for them.
Targeting those banks becomes a silent no-op instead of a hard error. If keeping
the error is preferred, that check can move above the gate — happy to adjust.

## Accuracy Tests

Not applicable: no change to kernels or model forward code. The change only
narrows which modules get wrapped, and only for models that already declare a
scope.

## Speed Tests and Profiling

Not applicable. Wrapping fewer modules cannot slow anything down; it avoids LoRA
slots for modules the adapter never fills.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

---

## 我们这边做过的验证

- `scripts/ci/check_registered_tests.py` 通过
- `isort` 7.0.0 / `black` 26.1.0（与上游 pre-commit 同版本）跑过，无改动
- 在 T4 容器里对这个分支跑 `pytest test/registered/unit/lora/test_should_apply_lora_gate.py`：
  - 带补丁：3 个用例通过
  - 手动删掉门的三行：立即失败（`modal_verify_pr1.py` 会把这两步都跑一遍）

## 没放进这个 PR 的东西

- `patch_torch` 的 CPU tensor 保护 —— 另开一个 PR，关注点不同
- `tp_worker` 补 `monkey_patch_torch_reductions` —— 上游 main 已经用统一的
  `_deserialize_own_rank` 修掉了，只有 v0.5.12.post1 还需要
- Qwen3-Omni 的 `_lora_pattern` 与音频对齐 —— 我们自己的 delta，等训练链路端到端跑通再说
