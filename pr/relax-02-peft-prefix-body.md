## What

`write_hf_peft_adapter` now writes adapter tensors under PEFT's `base_model.model.` key layout, via a small idempotent helper (`to_peft_state_dict`). The exported directory becomes a standard PEFT adapter rather than one that only looks like it.

## Why

`_save_lora_to_checkpoint` states the contract for the directory it produces:

> This is a portable *export* artifact (HF `adapter_model.safetensors` + `adapter_config.json`) for external / inference use — e.g. loading with `peft.PeftModel.from_pretrained`. It is NOT the resume source.

The docs repeat it in `docs/{en,zh}/guide/low-rank-adaptation-training.md`. The second sentence holds; the first does not.

`AutoBridge.export_adapter_weights` yields bare HF parameter names — `model.layers.0.self_attn.q_proj.lora_A.weight`, or `thinker.model.layers...` for a multimodal base such as Qwen3-Omni — and `write_hf_peft_adapter` hands them to `save_file` unchanged. PEFT, however, writes every key under the wrapper module it inserts, so its own files read `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.

The result is worse than a load error. Measured with `peft 0.20.0` / `transformers 5.6.0`, `PeftModel.from_pretrained` accepts the directory, emits a single `UserWarning: Found missing adapter keys`, and leaves every `lora_B` at zero — an adapter that loads successfully and changes nothing. A user who exports a trained adapter and evaluates it through PEFT measures the base model and has only a warning to explain the result.

The blast radius is narrow and worth stating explicitly:

| Path | Affected | Why |
| --- | --- | --- |
| Resume from checkpoint | No | LoRA params are ordinary model parameters in the native `torch_dist` checkpoint, exactly as documented |
| Adapter sync to SGLang (disk or tensor) | No | SGLang resolves adapter keys by suffix and a `layers.(\d+)` regex, so the prefix is immaterial |
| Loading the export with standard PEFT | **Yes** | Silently yields an all-zero adapter |

So the only behavior that changes is the one the artifact advertises and currently fails to deliver.

## How

Add `PEFT_STATE_DICT_PREFIX` and `to_peft_state_dict`, and apply the latter in `write_hf_peft_adapter` just before serialization.

The transform is idempotent: keys already carrying the prefix pass through untouched, so callers that hand over an exporter's output and callers that hand over an existing PEFT state dict converge on the same file. A submodel prefix such as Qwen3-Omni's `thinker.` is part of the parameter path and stays inside the PEFT prefix, matching what PEFT itself would write for that base model.

Scope was kept deliberately tight. The in-memory transport (`load_lora_adapter_from_tensors`) is untouched, since SGLang is prefix-agnostic and there is no reason to perturb the per-step sync path to fix an artifact contract. Normalizing at the single point where bytes hit disk covers both writers — the checkpoint export and the live adapter directory — without changing any caller.

## Testing

Run inside the Relax training image (`sglang 0.5.12.post1`, `megatron.bridge 0.5.0`, `torch 2.11.0+cu129`):

```
pytest tests/utils/test_megatron_peft_utils.py \
       tests/backends/megatron/weight_update/test_lora_weight_sync.py
48 passed, 5 skipped
```

Added `TestToPeftStateDict` (bare names, a multimodal `thinker.` path, idempotency, tensor identity) and a `write_hf_peft_adapter` case asserting the exact key set on disk for a bare-named export. The existing round-trip test already passes prefixed keys and still passes unchanged, which is the idempotency guarantee stated above.

`ruff format --check` and `ruff check` are clean on both touched files.

- [x] `pre-commit run --all-files` passes (ruff format + ruff check run directly, see above)
- [x] Tests pass (`pytest tests/`)
- [x] New tests added (if applicable)
- [ ] Documentation updated (if applicable)

## Type of Change

- [x] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] CI/CD or build changes

## Screenshots / Logs

Keys in `adapter_model.safetensors` for the same exported adapter:

```
# before
model.layers.0.self_attn.q_proj.lora_A.weight
model.layers.0.self_attn.q_proj.lora_B.weight

# after
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
```

Loading the "before" layout with PEFT, which is what the docstring recommends:

```
UserWarning: Found missing adapter keys while loading the checkpoint: [...]
# every lora_B stays at zero
```
