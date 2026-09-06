# PR 1 — sgl-project/sglang

Branch: `SakaiXue6666:fix/lora-honor-should-apply-lora` (based on main @ `f148eb6e6e`)

Link to open the PR (just needs a logged-in browser):
https://github.com/sgl-project/sglang/compare/main...SakaiXue6666:sglang:fix/lora-honor-should-apply-lora?expand=1

---

## Title

```
Honor should_apply_lora when wrapping LoRA target modules
```

---

## Body (paste as-is)

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

## What we verified on our side

- `scripts/ci/check_registered_tests.py` passes
- `isort` 7.0.0 / `black` 26.1.0 (same versions as the upstream pre-commit) run clean, no changes
- Ran `pytest test/registered/unit/lora/test_should_apply_lora_gate.py` on this branch in a T4 container:
  - with the patch: 3 tests pass
  - with the three gate lines manually removed: fails immediately (`modal_verify_pr1.py` runs both halves)

## What is deliberately not in this PR

- The CPU-tensor guard in `patch_torch` — separate PR, different concern
- Adding `monkey_patch_torch_reductions` to `tp_worker` — upstream main already fixed this
  with the unified `_deserialize_own_rank`; only v0.5.12.post1 still needs it
- Qwen3-Omni's `_lora_pattern` and audio alignment — our own delta; will revisit once the
  training path is proven end to end
