## What

`convert_megatron_to_hf_target_modules` now reduces a Bridge target-module *pattern* to the module name it selects before the Megatron→HF lookup, so patterns expand like bare names do. A pattern whose trailing segment is itself a wildcard is rejected instead of silently passed through.

## Why

Bridge's LoRA matcher accepts path patterns, and the tree already depends on that. `scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-lora-mtp-8xgpu.sh` uses them to scope adapters to the main decoder, and says why:

```sh
# Scoped to main transformer only. Bridge's LoRA matcher walks the full model
# (including `mtp.layers.*.mtp_model_layer.*`) — using bare short names like
# `linear_qkv` would also match MTP attention. The `*decoder.layers.*` prefix
# anchors matching to the main decoder path, so MTP layers stay frozen.
--lora-target-modules '*decoder.layers.*.linear_qkv' '*decoder.layers.*.linear_proj'
```

Injection honors the pattern; export does not. `convert_megatron_to_hf_target_modules` looks the whole string up in `MEGATRON_TO_HF_MODULES`, so a pattern misses and takes the "already HF-style or custom" passthrough branch. The glob then reaches every HF-facing consumer verbatim:

- `adapter_config.json`, both from `_save_lora_to_checkpoint` (which runs on every save whenever LoRA is enabled) and from `LoraAdapterSync.write_adapter_dir` for the live adapter directory
- the in-memory config handed to SGLang's `LoRAConfig.from_dict` in `LoraAdapterSync.config_dict`
- the SGLang server's `--lora-target-modules` in `sglang_engine`

None of these glob — PEFT and SGLang both match module names by suffix. So with the script above the exported adapter records `"target_modules": ["*decoder.layers.*.linear_qkv", "*decoder.layers.*.linear_proj"]` where it should record `["q_proj", "k_proj", "v_proj", "o_proj"]`, and nothing downstream matches a single module.

## How

Take the trailing dotted segment, then run the existing one-to-many expansion on it. Bare names contain no dot and are unchanged, so the common path keeps its current behavior; a fully qualified module path works like a pattern.

HF target module names carry no position, so a pattern's anchoring cannot be represented in the exported config. It does not need to be: the anchoring is already carried by which weights the exported adapter actually contains, which is what both PEFT and SGLang key off when they decide where an adapter applies.

The one case with no meaningful reduction is a wildcard inside the trailing segment itself (`*decoder.layers.*.linear_*`). Passing it through would reproduce exactly the failure above, so it raises `ValueError` naming the offending pattern and pointing at the anchored form.

## Testing

Run inside the Relax training image (`sglang 0.5.12.post1`, `megatron.bridge 0.5.0`, `torch 2.11.0+cu129`):

```
pytest tests/utils/test_megatron_peft_utils.py \
       tests/backends/megatron/weight_update/test_lora_weight_sync.py
47 passed, 5 skipped
```

Four cases were added to `TestConvertMegatronToHfTargetModules`, covering the pattern from the shipped script, de-duplication between a pattern and the bare name it ends with, a fully qualified path, and the rejected trailing wildcard. Reverting only `relax/utils/megatron_peft_utils.py` to `main` and rerunning those four fails all four, so they do pin the change.

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

Behavior of the conversion before and after, for the arguments the shipped MTP script passes:

```python
# before
convert_megatron_to_hf_target_modules(["*decoder.layers.*.linear_qkv", "*decoder.layers.*.linear_proj"])
['*decoder.layers.*.linear_qkv', '*decoder.layers.*.linear_proj']

# after
['q_proj', 'k_proj', 'v_proj', 'o_proj']
```
