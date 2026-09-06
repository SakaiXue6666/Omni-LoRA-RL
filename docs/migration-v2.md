# The v2 workspace — migrating to Relax's official LoRA adapter mode

**Goal**: move Qwen3-Omni Thinker's LoRA reinforcement learning from v1's home-grown direct
weight-export route to the LoRA adapter mode Relax already supports officially.

**Scope**: Thinker LoRA + S2TT text output only. The Talker / speech path is frozen wholesale in
v1 and does not come into v2.

**Verification platform**: Modal.

## Relation to v1

v1 is not being replaced; it is frozen into a **parity oracle**: every step of v2's behavior can
be checked against v1's known results.

| | v1 | v2 |
|---|---|---|
| Directory | `d:\Li_Lab\RL\omni-lora-rl` | `d:\Li_Lab\RL\omni-lora-rl-v2` (this directory) |
| Hub branch | `sglang_omni` @ `fd7915b`, tag `v1-frozen` | `v2` (based on `main`) |
| Submodules | Relax + sglang + sglang-omni | Relax + sglang |
| Image | `slimerl/slime@sha256:bd219aba…` (= `nightly-dev-20260428a`) | `ghcr.io/redai-infra/relaxrl@sha256:8dc39af3…` |
| LoRA weight path | home-grown direct export | Relax's official adapter mode |

All of v1's details are in `README_v1.md` on the `v1` branch, and the code delta is additionally
archived as three patches in `patches/` on the `v1` branch.

This directory still keeps a `v1local` remote pointing at the local v1 repository, so
`git fetch v1local` can retrieve any v1 commit at any time.

## Why it is based on the main branch

That is how the GitHub repository was already split: `main` = Relax + sglang, `sglang_omni` =
Relax + sglang + sglang-omni. v2's scope matches `main`'s structure exactly, so it starts from
`main` rather than stripping a submodule off `sglang_omni`.

Two historical training-side records stay on the `sglang_omni` branch:
`IMPORTANT/OMNI_RELAX_HANDOFF_2026-07-18.md` and `IMPORTANT/OMNI_STREAMING_GPU_2026-07-19.md`.
The speech scripts and v1's frozen artifacts all stay in v1 and are not brought along.

## Submodule baselines

Both submodules are pinned to **recent upstream**, so our changes can be filed as upstream PRs
directly later on.

| Submodule | Upstream | v2 baseline | v2 branch |
|---|---|---|---|
| Relax | `redai-infra/Relax` | `main` @ `9a5674af` | `lora-omni-v2` |
| sglang | `sgl-project/sglang` | tag `v0.5.12.post1` (`5a15cde858`) | `lora-omni-v2` |

The sglang version cannot be picked freely: Relax's Dockerfile has
`BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-cu129`, and Relax **patches sglang itself** —
`docker/patch/latest/sglang.patch` is a symlink to `docker/patch/sglang/v0.5.12.post1.patch`
(135 KB, touching 44 files), applied at build time to `/sgl-workspace/sglang` inside the image.
Pick another version and that patch no longer applies.

### The layering convention for sglang

| Layer | Contents | Commit |
|---|---|---|
| 0 | Upstream tag `v0.5.12.post1` | `5a15cde858` |
| 1 | Relax's `v0.5.12.post1.patch` (44 files, vendored verbatim) | `ce1717a786` |
| 2+ | Our Qwen3-Omni LoRA delta | to be ported |

This way the whole submodule can be mounted into the container without losing Relax's changes,
and when filing a PR later we take only the diff from layer 2 onward, which stays clean against
upstream.

v1's delta is 7 files in total and overlaps Relax's 44 files in exactly one place,
`python/sglang/srt/managers/tp_worker.py`. The core files — `models/qwen3_omni_moe.py`,
`lora/lora_manager.py`, `utils/patch_torch.py` — are untouched by the upstream patch, so the
layering is clean.

v1's sglang state is also kept locally: branch `lora-omni-baseline`, tag `v1-frozen`
(`d13903a9`), available for direct comparison while porting.

## Environment validation (done, 2026-08-10)

On `ghcr.io/redai-infra/relaxrl@sha256:8dc39af3…` all four Phase 0.1 criteria passed, plus 0.1b
(re-checking recognition after registering the Qwen3-Omni bridge):

- `megatron.core` and `megatron.bridge` coexist
- `AutoBridge` has the adapter export API
- PEFT LoRA imports
- `AutoBridge` recognizes Qwen3-Omni

This is exactly where the 2026-07 migration attempt got stuck — back then Megatron-Bridge 0.5.0
and megatron-core could not be installed together. The newer Relax Dockerfile solves it with a
`3rdparty/Megatron-LM` submodule plus rsync, so this route is now open.

Related scripts: `scripts/probes/mig_00_env.py` (criteria 1–4),
`scripts/probes/mig_01_omni_bridge.py` (re-check after registration),
`scripts/probes/modal_env_gate_v2.py`, `scripts/probes/modal_gate_omni.py`.

The probes use "prebuilt image + specified source": the image is the official one, while the Relax
source is injected via `PYTHONPATH`, which saves rebuilding the image for every change.

## The key differences from v1 to v2

| How v1 did it | The official Relax facility in v2 |
|---|---|
| Hand-written direct weight export | `relax/backends/megatron/weight_update/lora_adapter_sync.py` |
| Hand-rolled LoRA injection | `apply_lora_to_model` in `relax/utils/megatron_peft_utils.py` |
| Hand-rolled adapter conversion | Megatron-Bridge's `export_adapter_weights` |
| — | Reference script `Relax/scripts/training/text/run-qwen3-4B-lora-adapter-x8gpu-async.sh` |
| — | Reference test `tests/backends/megatron/weight_update/test_lora_weight_sync.py` |

The sglang side is different: upstream still has no LoRA support for Omni — `qwen3_omni_moe.py`
has neither `should_apply_lora` nor any logic excluding the audio/vision towers from LoRA. That
part is a **permanent delta**, not technical debt; it has to be carried indefinitely (and it is
exactly what a future PR to sgl-project would contain).

## Probe 1 conclusion: the scope of LoRA (verified, 2026-08-11)

Scripts `scripts/probes/mig_02_lora_scope.py` + `scripts/probes/modal_probe_lora_scope.py`, run on
the v2 image with Relax `9a5674af` (T4, about 2 minutes; the towers are built on the meta device,
no weights loaded).

Relax's Megatron model `Qwen3OmniMoeModel` really does build both towers on the `pre_process`
rank, and it uses **the transformers HF implementations** (`Qwen3OmniMoeAudioEncoder` /
`Qwen3OmniMoeVisionEncoder`); only `language_model` is Megatron's GPT. `PEFT.__call__` goes
through the generic `_walk_model`, which descends into the HF submodules, so the towers are within
traversal range — whether LoRA attaches comes down entirely to whether the names collide.

Measured (transformers 5.6.0):

| Tower | Linear leaf names | Collides with Megatron naming? |
|---|---|---|
| audio | `q_proj` `k_proj` `v_proj` `out_proj` `fc1` `fc2` `proj1` `proj2` `conv_out` | none collide |
| vision attention / merger | `qkv` `proj` `0` `2` | no collision |
| vision MLP | `linear_fc1` `linear_fc2` | **collides** |

Match results:

- `--lora-target-modules linear_qkv linear_proj` (v1's and the official default, i.e. Q/K/V/O) →
  **zero modules matched** in either tower; LoRA lands only on the language model.
- Adding `linear_fc1 linear_fc2` → matches `vision_model.blocks.N.mlp.linear_fc1/2`, i.e. the MLP
  of every vision-tower layer (the probe squashes depth to 1; at the real 27 layers that would be
  54 modules).

**Conclusion: attaching only the attention projections requires no change to Relax's injection
logic — the CLI default is correct.** But this is an implicit constraint: the day someone wants
LoRA on the MLPs, they must first exclude the vision tower with a wildcard (`ModuleMatcher`
supports patterns like `*.layers.0.*.linear_qkv`) or with `exclude_modules`, or adapters get
silently attached to the vision tower while the SGLang base has no corresponding modules at all.

## Probe 2 conclusion: export naming and parity with SGLang (verified, 2026-08-11)

Scripts `scripts/probes/mig_03_export_names.py` + `scripts/probes/modal_probe_export_names.py`
(T4, about 2 minutes; again without building a Megatron model — the naming is derived from the
real `mapping_registry`, and the SGLang side is fed zero tensors shaped from the real config).

It also confirmed that the prebuilt image is not behind Relax `main`'s pins: `sglang 0.5.12.post1`,
`megatron.bridge 0.5.0`, `megatron.core 0.18.0`, `transformers 5.6.0`, `torch 2.11.0+cu129`.

On the export side the naming rule is `linear_in → lora_A`, `linear_out → lora_B`, while the
adapter's HF name is derived from **the base weight's mapping** (`_resolve_hf_adapter_param_name`
→ `mapping_registry.megatron_to_hf_lookup` → `_make_lora_param_name`), so Omni's `thinker.`
prefix comes along automatically. Measured:

| Megatron side | Exported HF name |
|---|---|
| `...self_attention.linear_qkv.adapter.linear_in/out` | `thinker.model.layers.N.self_attn.{q,k,v}_proj.lora_{A,B}.weight` |
| `...self_attention.linear_proj.adapter.linear_in/out` | `thinker.model.layers.N.self_attn.o_proj.lora_{A,B}.weight` |

The fused QKV `linear_out` is split into q/k/v by upstream's `_split_qkv_linear_out_weight`
calling `split_qkv_weights(model.config, ...)`, using the same de-interleaving logic as the base
weights.

Round-trip verification on the SGLang side (30B-A3B thinker: hidden=2048, 32 heads, 4 KV groups,
head_dim=128, taking rank=32):

- `get_layer_id` is `re.search(r"layers\.(\d+)\.")`, prefix-independent, so names carrying
  `thinker.` still parse out the layer number
- `normalize_qkv_proj` stacks the split q/k/v back into `qkv_proj`: `lora_B` comes out as
  `(5120, 32)` = `32*128 + 2*4*128`, `lora_A` as `(96, 2048)` = `3*32`, matching the buffer the
  memory pool allocates as `max_lora_dim * 3`

**Three differences from v1 and their consequences:**

1. **Split vs fused** — v1's hand-written export produced `qkv_proj.lora_{A,B}` directly; v2
   produces separate q/k/v, and SGLang's `normalize_qkv_proj` stacks them back. Equivalent.
2. **QKV de-interleaving** — v1 wrote `_reorder_qkv_lora_b` to manually rearrange the
   query-group-interleaved layout into `[q;k;v]`; upstream already does the same thing.
   **That whole block of code can be discarded.**
3. **The `base_model.model.` prefix** — v1's names carry this prefix, v2's do not (upstream only
   adds it in `convert_adapter_weights_to_peft_state` when writing to disk, and Relax's
   `export_local_adapter` / `write_hf_peft_adapter` both use `param_name` verbatim without going
   through that function). No impact on the path that pushes to SGLang, since SGLang matches
   entirely on suffixes and regexes; but it does affect the exported artifact — see the next
   section.

## Probe 3 conclusion: the exported adapter directory cannot be read back by stock PEFT (verified, 2026-08-11)

Scripts `scripts/probes/mig_04_peft_prefix.py` + `scripts/probes/modal_probe_peft_prefix.py`
(pure CPU, about 2 minutes).

It came out of probe 2 noticing that the keys Relax writes to disk lack the `base_model.model.`
prefix — `write_hf_peft_adapter` simply `save_file`s `export_adapter_weights`'s `param_name`
verbatim, without going through upstream's `convert_adapter_weights_to_peft_state` (which is the
function that adds the prefix).

First, clearing up a **worry that does not hold**: this does not affect resuming from a
checkpoint. `_save_lora_to_checkpoint` in `checkpoint.py` is explicit that `lora_adapter/` is a
"portable export artifact for external/inference use" and **not a resume source** — LoRA
parameters are ordinary model parameters, and the native Megatron torch_dist checkpoint already
holds them.

But the claim "for example, load it with `peft.PeftModel.from_pretrained`" does not hold.
Measured (peft 0.20.0 + transformers 5.6.0): the keys stock PEFT writes are all of the form
`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`; strip the prefix and read it
back, and `PeftModel.from_pretrained` **does not error** — it emits a single
`UserWarning: Found missing adapter keys` and then `lora_B` is all zeros, i.e. you get a model
that learned nothing.

Scope of impact:

| Path | Affected? | Why |
|---|---|---|
| Pushing to SGLang during training (memory / disk) | No | SGLang matches on suffix and regex, prefix-independent |
| Resuming from checkpoint | No | Goes through the native torch_dist checkpoint |
| Loading the exported adapter with stock PEFT | **Yes** | Silently loads an all-zero LoRA |

It also incidentally confirms that v1's `base_model.model.thinker.model.layers...` naming was
PEFT-compliant all along.

## Porting the sglang delta (done, 2026-08-11)

v1's delta was written against 0.5.9 and consists of five changes. Checking each one against
v0.5.12.post1 (plus Relax's vendor patch), **four are still needed**:

| v1's change | Status on v0.5.12.post1 |
|---|---|
| The `getattr` compatibility shim in `lora_manager` | Dropped — all four ServerArgs fields now exist with matching defaults |
| The `should_apply_lora` gate in `lora_manager` | Still needed, and it is an upstream regression (see below) |
| Adding `monkey_patch_torch_reductions` to `tp_worker` | Still needed — upstream only calls it inside `update_weights_from_tensor` |
| The CPU-tensor guard in `patch_torch` | Still needed — it still rewrites argument 7 unconditionally |
| Audio alignment + LoRA declaration in `qwen3_omni_moe` | Still needed — upstream leaves that file untouched |

**`should_apply_lora` is an upstream regression**: in v0.5.12.post1 thirteen models define the
hook (`qwen3_vl`, `qwen3_vl_moe`, `qwen2_vl`, `gpt_oss`, `gemma*` and others), and the comment in
`init_lora_modules` still says "embed_tokens and lm_head must be handled before the
should_apply_lora gate", but **there is no call site anywhere in the repository** — the hook is
dead code. Relax's vendor patch never touches `lora_manager.py`, so this is upstream's own state.
In other words, what v1 wrote back then was not an Omni-specific hack; it was filling in behavior
upstream itself assumes exists.

The composition of the `lora-omni-v2` branch after porting (three commits stacked on upstream tag
`v0.5.12.post1`):

1. `ce1717a786` — Relax's vendor patch (44 files), keeping parity with the official Docker image
2. `b7225ce5ec` — three upstream fixes: restore the gate call, guard the CPU reducer, install the
   torch reducer before loading an adapter
3. `aa87211755` — Omni-specific: declare LoRA support on the outer class + variable-length audio
   alignment

Splitting into two commits is so that the second can later be lifted out and filed with
sgl-project without having to disentangle it from the Omni changes.

## Probe 4 conclusion: which modules `_lora_pattern` actually hits (verified, 2026-08-11)

The first three probes were all training-side (Megatron / Bridge); this one is inference-side:
actually build sglang's `Qwen3OmniMoeForConditionalGeneration` and see what `_lora_pattern`
matches against **real module names**.

Approach: shrink the config only (text 48→2 layers, experts 128→4, audio 32→2, vision 27→2), build
on the meta device, load no weights. Two minutes on a single T4. `initialize_dp_attention` must be
called before building, otherwise `LayerCommunicator` raises `dp attention not initialized` while
constructing a decoder layer.

Out of 121 modules, `--lora-target-modules qkv_proj o_proj` **matches 8 by pure suffix, 4 of them
inside the towers**:

```
thinker.visual.blocks.0.attn.qkv_proj              <-- blocked by the gate
thinker.visual.blocks.1.attn.qkv_proj              <-- blocked by the gate
thinker.audio_tower.layers.0.self_attn.qkv_proj    <-- blocked by the gate
thinker.audio_tower.layers.1.self_attn.qkv_proj    <-- blocked by the gate
thinker.model.layers.{0,1}.self_attn.{qkv_proj,o_proj}   <-- allowed through
```

In other words, without the `should_apply_lora` gate, upstream attaches **half** of the target
modules to the vision and audio towers — modules the adapter has no weights for at all. This is
precisely the inference-side mirror of probe 1 (training only attaches LoRA to the language
model); the two sides close the loop.

It also confirms that of the 8 modules the gate itself lets through (`embed_tokens`, `lm_head`,
each layer's `mlp.experts` and the two projections), only the projections actually land inside the
target set; the rest are placeholders in the pattern and cause no collateral damage.

Scripts: `scripts/probes/mig_05_sglang_lora_scope.py` +
`scripts/probes/modal_probe_sglang_scope.py`. Note that the runner clones the fork's
`lora-omni-v2` branch and puts `python/` at the front of `PYTHONPATH`; otherwise you are testing
the sglang preinstalled in the image and will not see the newly added gate.

## Unit tests (done, 2026-08-11)

v1's `test_should_apply_lora_gate.py` lived under `test/srt/lora/` in pytest style. 0.5.12.post1
reorganized tests into `test/registered/` (service-running) and `test/registered/unit/`
(non-service), with requirements: mirror the source tree's directory structure, call
`register_cpu_ci` / `register_cuda_ci` at the top of the file, use `unittest` + `CustomTestCase`,
and never call `pytest.main` bare under `__main__` (the repo has `test_no_bare_pytest_main.py`
specifically to check this). So it was rewritten to the new convention and split in two:

| File | Contents | Ownership |
|---|---|---|
| `test/registered/unit/lora/test_should_apply_lora_gate.py` | Generic gate behavior: towers stay unwrapped, models without the hook keep suffix matching, a deny-all hook wraps nothing | Filed with the upstream PR, free of any Omni dependency |
| `test/registered/unit/models/test_qwen3_omni_lora_pattern.py` | Positive and negative examples for `_lora_pattern`, with module names taken from probe 4's measurements | Our delta |

Splitting them lets the upstream PR carry only the generic test without dragging the Omni changes
along. Both rely on `LoRAManager.__new__` to skip `__init__`, touching neither the memory pool nor
any adapter download. To run: `modal run modal_run_lora_tests.py` (mounts the local test files onto
a clone of the fork, so changes do not have to be pushed first).

## Upstream PRs (from 2026-08-11)

Five changes were confirmed to be upstream problems (rather than our adaptation), and each was
filed as a PR. Authorship is mine alone.
The statuses below are as of 2026-09-05; the README's "Current status" is the source of truth.

| PR | Repo | Branch | Status |
|---|---|---|---|
| Honor `should_apply_lora` when wrapping LoRA target modules | sgl-project/sglang | `fix/lora-honor-should-apply-lora` | Filed as #34428; CI blocked on the missing `run-ci` label, tests have never actually executed |
| `fix(lora): expand path-pattern target modules to HF names` | redai-infra/Relax | `fix/lora-target-modules-wildcard-export` | Filed as #261, CI green |
| `fix(lora): write exported adapters in PEFT's key layout` | redai-infra/Relax | `fix/lora-adapter-peft-prefix` | Filed as #262, CI green after fixing docformatter |
| `fix(lora): inline adapter tensors into the engine payload` | redai-infra/Relax | `fix/lora-adapter-transport-shm` | Filed as #265 (2026-08-12), awaiting code owner |
| Fix IndexError when reducing CPU tensors after `monkey_patch_torch_reductions` | sgl-project/sglang | `fix/reduce-tensor-cpu-guard` | Filed as #34595 (2026-08-12), CI blocked on the missing `run-ci` label |

The bodies are in `docs/upstream-prs/sglang-01-body.md`,
`docs/upstream-prs/relax-01-wildcard-body.md`, `docs/upstream-prs/relax-02-peft-prefix-body.md`
and `docs/upstream-prs/relax-03-transport-body.md`.

The two Relax ones each have hard evidence behind them:

- **Wildcards**: Relax's own
  `Relax/scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-lora-mtp-8xgpu.sh` uses
  `*decoder.layers.*.linear_qkv`, with a comment saying it is there to keep the MTP layers frozen.
  The injection side (Bridge) understands that pattern; the export side does not — the glob lands
  verbatim in `adapter_config.json` and in the SGLang launch arguments, and since both match only
  on suffixes, the exported config hits exactly zero modules.
- **PEFT prefix**: the docstring of `_save_lora_to_checkpoint` and both the Chinese and English
  docs promise that `lora_adapter/` can be loaded with `peft.PeftModel.from_pretrained`, but the
  keys written to disk lack `base_model.model.`. Probe 3 measured it: no error, just one warning
  about missing keys, and then `lora_B` is all zeros. Resuming and SGLang are both unaffected;
  what is affected is precisely the one use this artifact promises.

- **Adapter transport**: on real hardware with 4×A100, the very first adapter push died on TP0
  while TP1–3 succeeded. The root cause is that the payload carries a `/dev/shm` reference rather
  than bytes — see probe 9 below. This PR extracts serialization into
  `megatron_peft_utils.serialize_adapter_tensors` and then switches it to inlining. Extracting the
  function is not cosmetic: the original call site lives in `update_weight_from_tensor.py`, which
  does `from megatron.core import mpu` at module level, so it cannot be imported in CPU CI and the
  test could only be skipped — no protection at all. Moved into a pure-torch utility module, the
  test lands in `tests/utils/test_megatron_peft_utils.py`, which already runs in CI.

- **CPU tensor out of range**: `monkey_patch_torch_reductions()` replaces the reducer for **all**
  tensors, not just CUDA ones, while `_reduce_tensor_modified` rewrites argument 6
  unconditionally — the tuple a CPU tensor reduces to simply does not have that slot, hence
  `IndexError`. This one is not unique to us: verl's
  [#4065](https://github.com/volcengine/verl/issues/4065) has been open since November 2025, with
  multiple reproductions and an identical traceback, and the workaround circulating in that thread
  is exactly this length guard. Which means downstream users today either hand-edit
  site-packages or switch to merge mode to avoid pushing adapters.

One CI pitfall we hit: Relax's `.pre-commit-config.yaml` has a local hook
`docformatter --wrap-descriptions 79` that ruff knows nothing about. The first revision of #262
failed because a docstring in a test file was wrapped at ~90 columns; docformatter rewrote it and
the check reported "files were modified" (Lint and ruff both passed, so running ruff alone shows
nothing). Downloading the logs requires being logged in; running the repo's own pre-commit
verbatim inside a container via `scripts/probes/modal_precommit_relax_prs.py` reproduces it. From
now on, when filing PRs to Relax, wrap docstring description paragraphs to 79 columns or just run
that script.
Both branches were verified two ways (`scripts/probes/modal_verify_relax_prs.py`): with the patch,
47/48 tests pass; swap `megatron_peft_utils.py` back to upstream `main` and rerun, and every newly
added test fails. `ruff format --check` and `ruff check` are clean
(`scripts/probes/modal_lint_relax_prs.py`, run in a container because the local machine cannot
reach the pip index).

## Probe 6: CPU tensor reduce out of range (2026-08-11, pure CPU)

Evidence gathered for the third sglang PR, which also explains why the `patch_torch` and
`tp_worker` changes have to be filed together.

Upstream's `_reduce_tensor_modified` rewrites index 6 of the argument tuple unconditionally, with
a comment justifying it as "the signature has not changed in years". That assumption only holds
for CUDA tensors. Measured, a CPU tensor coming out of `reductions.reduce_tensor` is:

```
rebuild function: rebuild_tensor      argument tuple length: 3
  [0] _TensorMeta   [1] TypedStorage   [2] (0, torch.Size([4, 4]), (4, 1), False)
```

Length 3, so indexing 6 raises `IndexError: tuple index out of range` outright — not a silent
corruption of some field, a hard crash. All three A/B legs hold: the guarded version serializes
fine, swapping in upstream's version throws IndexError on the spot, swapping back to the guarded
version passes again.

The causal chain is worth spelling out, otherwise it is easy to mistake this for a pre-existing
upstream bug: v1 pushed CPU tensors on 0.5.9 without trouble because the LoRA path never called
`monkey_patch_torch_reductions` back then; the out-of-range access only surfaced after *we* added
the reducer installation to `tp_worker`. So the two changes are two halves of the same thing and
must be filed together. verl hit the same pitfall (bug #4065).

Script: `scripts/probes/mig_07_cpu_reduce.py`, entry point
`modal run modal_probe_serve_lora.py --stage reduce` (CPU, about 90 seconds).

## Probe 5: hot-pushing an adapter to sglang on real hardware (2026-08-11, 1×A100-80GB)

The first time v2's inference side ran on real hardware. The criteria follow v1's three — judging
only "the output changed" is too weak, since changing one byte of a weight also changes the
output.

| Check | Result |
|---|---|
| Start Omni with `enable_lora` | It starts. This step is itself a validation of the gate: if the towers were wrongly wrapped, the hidden dim would not line up at load time |
| Base generation | `今天天气很好。` ("the weather is nice today"), with the chat template applied (v1's lesson: without it you get `<|im_end|>` straight away) |
| Reversibility: push an adapter with B=0 | **Maximum logprob difference 0.000000** against base |
| Effectiveness: swap in a non-zero B (scale 0.12) | Maximum difference 2.416, output visibly diverges |
| Stability: three consecutive unload → load rounds | No crash; free memory stable at 11.5/79.3 GiB |

Reversibility is the most valuable line here: reproducing base exactly at B=0 means all 384 tensor
names were accepted by sglang. If some name did not line up, that portion of the weights would be
silently dropped and B=0 would look identical anyway — but conversely, exact reproduction combined
with a non-zero B genuinely taking effect is what rules out both "the names were wrong so nothing
applied" and "the names were wrong but it happened not to matter". This upgrades probe 2's static
name comparison into a real-hardware verification.

The adapter is CPU tensors throughout (384 of them), taking exactly the serialization path from
probe 6.

Script: `scripts/probes/mig_06_serve_lora.py`, with three stages: `--stage inspect` (CPU, inspect
the model volume and source), `--stage reduce` (CPU, probe 6), `--stage probe` (A100, this
section). The design follows experiments G/J of `modal_run.py` on the `v1` branch directly:
engine parameters `disable_cuda_graph=True, mem_fraction_static=0.85, tp_size=1`, the chat
template, how the adapter is constructed, the logprob comparison — all already validated in v1, so
there was no re-derivation this time.

## Probe 7: training-side smoke test (2026-08-12, 1×A10G, about 3 minutes)

The first six probes all validated inference-side behavior and static predictions about export
naming; the training side had never run on v2 at all. This step answers exactly three things and
touches neither rollout, dataset, nor optimizer.

The key to keeping it cheap is shrinking layers: `provider.num_layers = 2`, randomly initialized,
without loading the real 30B weights. This is not a hack we invented — Relax's `model_provider.py`
lists `num_layers` in `bridge_keys` with a comment saying "Allow CLI to override layer count for
layer-reduced training"; it is officially supported. Shrunk down to 3.06B, one A10G (22 GiB) is
more than enough — no A100 needed.

| Check | Result |
|---|---|
| `AutoBridge` builds Omni | `Qwen3OmniModelProvider` → `Qwen3OmniMoeModel`, 48 layers shrunk to 2 |
| LoRA injection scope | 8 adapter parameters, 4 per layer (A/B for each of `linear_qkv` / `linear_proj`), all under `language_model.*`, none in the towers |
| Base model frozen | The trainable parameters are exactly those 8 adapters |
| Adapter export | 16 tensors; naming and shapes below |

Exported naming and shapes:

```
thinker.model.layers.{0,1}.self_attn.q_proj.lora_A.weight  (32, 2048)
thinker.model.layers.{0,1}.self_attn.q_proj.lora_B.weight  (4096, 32)
thinker.model.layers.{0,1}.self_attn.k_proj.lora_B.weight  (512, 32)
thinker.model.layers.{0,1}.self_attn.o_proj.lora_A.weight  (32, 4096)
```

Three conclusions:

1. **The Megatron side is a fused `linear_qkv`, and on export Bridge's `QKVMapping` splits it into
   `q/k/v_proj`.** That is exactly what v1's hand-written de-interleaving did, and upstream now
   does it — probe 2 inferred it statically, this is empirical on a real model. That code can
   retire.
2. **The shapes match v1's hard-coded values one by one**: hidden 2048, q_out 4096, kv_out 512.
3. **The counts close the loop**: 2 layers export 16, so 48 layers is 384 — exactly the 384
   tensors probe 5 pushed to the sglang engine. What the training side produces and what the
   inference side consumes mesh.

`convert_megatron_to_hf_target_modules(['linear_qkv', 'linear_proj'])` lands in
`adapter_config.json` as `['q_proj', 'k_proj', 'v_proj', 'o_proj']`, fully covering the exported
leaf modules.

Scripts: `scripts/probes/mig_08_train_side.py` + `scripts/probes/modal_probe_train_side.py`. They
use Relax's own `build_lora_peft` and bridge rather than reimplementing them, so what is being
validated is the real code path.

## Comparison against v1 (2026-08-12, a check before end-to-end training)

Going through v1's training side and comparing item by item against probe 7's results, five things
are worth recording.

**1. The LoRA hyperparameters differ and must be reverted to v1's for the end-to-end run.** v1's
smoke script used `--lora-rank 16 --lora-alpha 32` (with `--sglang-max-lora-rank 16` on the sglang
side); probes 5 and 7 used 32/64. Both work functionally, but v1's 100-step S2TT BLEU curve was
produced at 16/32, so comparing against it requires matching hyperparameters — otherwise there is
no telling whether a difference comes from the migration or from the rank.

**2. The `--lora-target-modules` default is a v1 delta that we did not port, and should not.** v1
changed the default to `*language_model*linear_qkv` / `*language_model*linear_proj`, on the
grounds that a bare `linear_qkv` would attach to `audio_model`. But that v1 conclusion came from a
hand-built Omni-like tree in `verify_lora_attach.py` on the `v1` branch, not a real model. Probe 7
used bare names on the `Qwen3OmniMoeModel` Relax actually builds, and all 8 adapter parameters
landed in the language model with the towers clean — because v2's audio/vision are HF modules
named `q_proj/k_proj/v_proj`, nothing called `linear_qkv` at all.

   There is also a reverse dependency here: **the wildcard is precisely the bug we filed
   [#261](https://github.com/redai-infra/Relax/pull/261) against Relax to fix** (the glob lands
   verbatim in `adapter_config.json` and the export side matches zero modules). So actually using
   v1's hardened form would require pulling in #261 first; using bare names does not. The
   conclusion is to keep bare names and not make #261 a prerequisite for end-to-end training.

**3. v1's guard that warns when an adapter name contains audio/vision (`_assert_lora_attached`)
does not exist in v2 upstream.** Probe 7 shows it is not currently needed, but it is cheap and
could become a small follow-up PR to Relax, or just an assertion kept on our side for now.

**4. v1 has no `export_adapter_weights` at all.** The redai bridge it pinned (`f13bec09`) predates
that API, so export went through the direct route in `convert_qwen3omni_to_hf`, which additionally
required a hand-written `_reorder_qkv_lora_b()` to do the GQA dimension rearrangement. Probe 7
shows v2's bridge natively splits fused `linear_qkv` into `q/k/v_proj` — so that entire direct
path, rearrangement function and all, gets deleted in v2. This is the single biggest subtraction
in the migration.

**5. The next risk point is adapter gather at TP>1; probe 7 only covered TP=1.** v1 bled here
(pitfall 16: a hand-written all_gather at TP=4 hit a CUDA illegal access) and deliberately kept
two probes on the `v1` branch, `verify_lora_tp.py` (TP=2) and `verify_tp_gather.py` (TP=4). v2
hands the gather to bridge and it will probably be fine, but "probably" is not verified — it is
worth spending two cards on an export-parity check before going to 4×A100.

The v1 configuration to copy for the end-to-end run: `TP=4 / EP=4 / PP=1`, **sequence-parallel and
recompute off** (v1's record: recompute + SP + LoRA makes `lora_B`'s backward produce NaN),
`A100-80GB:4`, 240-minute timeout, `retries=10` (resume from the checkpoint on the volume after
Modal preempts).

## Probe 8: adapter export parity at TP=2 (2026-08-12, 2×A10G, about 5 minutes)

Probe 7 only covered TP=1, and TP>1 adapter gather is exactly where v1 bled (pitfall 16: a
hand-written all_gather at TP=4 hit a CUDA illegal access), which is why two dedicated probes were
kept back then. v2 hands the gather to bridge, so it ought to be fine — but "ought to" is not
verified.

There is one trap in designing the criteria that has to be avoided: **you cannot compare TP=1 and
TP=2 exports numerically**. Megatron seeds its RNG per rank at TP initialization, so the same
logical weight is simply not the same set of random numbers at two different degrees of
parallelism, and any measured difference is meaningless. Instead, two self-consistent criteria:
the shapes must be full size; and concatenating the exported q/k/v blocks must yield the same set
of rows as the fused matrix we `all_gather` by hand.

Everything passed, and along the way we got a clear look at the sharding — the most valuable
finding of this round:

```
linear_qkv.adapter.linear_in   (16, 2048)   partition_dim=0   <-- the rank dimension is sharded!  32 -> 16
linear_qkv.adapter.linear_out  (2560, 32)   partition_dim=0        output dimension sharded, 5120 -> 2560
linear_proj.adapter.linear_in  (32, 2048)   partition_dim=1        input dimension sharded
linear_proj.adapter.linear_out (1024, 32)   partition_dim=0
```

**The LoRA rank dimension itself participates in TP sharding** (`linear_in` has only 16 rows per
card; the two cards together make rank=32), which we had not realized. The exported
`q_proj.lora_A` is a complete `(32, 2048)`, meaning bridge stitches that dimension back correctly
too. Three different sharding patterns (rank dimension, output dimension, input dimension) are all
reconstructed correctly in a single export.

Reconciliation details: the hand-gathered fused `linear_out` is `(5120, 32)`, and the exported
q+k+v concatenation is also `(5120, 32)`; sorted, the two are element-wise equal. Every row of
`q_proj` can be found in the fused matrix — the gather lost no data and the GQA de-interleaving is
not misaligned.

One observation: at TP>1, `Qwen3OmniModelProvider.finalize()` force-enables `sequence_parallel`.
This run did no backward pass so it did not matter, but v1 recorded that recompute + SP + LoRA
makes `lora_B`'s backward produce NaN, so watch that interaction in the end-to-end run.

Scripts: `scripts/probes/mig_09_tp_export.py` + `scripts/probes/modal_probe_tp_export.py`
(torchrun with 2 processes).

## To do

- [x] Probe 1: LoRA scope — see above
- [x] Probe 2: parity between `export_adapter_weights` naming and v1's direct export — see above
- [x] Probe 3: whether the exported adapter directory can be read back by stock PEFT — it cannot,
      see above
- [x] Port v1's sglang delta to `v0.5.12.post1` — see above
- [x] Probe 4: what `_lora_pattern` matches against real module names — see above
- [x] File a PR with sgl-project: restore the `should_apply_lora` call site — #34428
- [x] Add unit tests for the gate — rewritten to upstream's new directory convention, 6 tests all
      pass on a T4 — see above
- [x] File the first Relax PR: `convert_megatron_to_hf_target_modules` support for path patterns —
      branch pushed, body written
- [x] File the second Relax PR: `write_hf_peft_adapter` adding the `base_model.model.` prefix —
      branch pushed, body written
- [x] Probe 6: A/B on the CPU tensor reduce out of range — upstream IndexError, guarded version
      passes, see above
- [x] Probe 5: verify adapter hot-loading, reversibility and stability on 1×A100 — all five checks
      pass, see above
- [x] The CPU-tensor out-of-range guard in sglang's `patch_torch` — branch pushed, body written;
      the `tp_worker` part is obsolete on main (upstream consolidated deserialization into
      `_deserialize_own_rank`, which installs the reducer itself, and the LoRA path goes through it
      too)
- [x] Probe 7: training-side smoke test (Bridge builds Omni + LoRA injection + adapter export) —
      passed first try, see above
- [x] Probe 8: adapter export parity at TP=2 — all three sharding patterns reconstructed
      correctly, see above
- [x] Probe 9: three-way comparison of adapter transport — reproduced the real-hardware ENOENT,
      inlined bytes pass, see below
- [x] File the third Relax PR: switch adapter pushing to inlined bytes — branch pushed, body
      written
- [x] Get end-to-end training running on real hardware — 40 steps of S2TT on 4×A100, tracking v1's
      curve segment by segment, see below
- [ ] Port simultaneous S2TT — v1's `examples/simul_s2tt/rollout.py` goes through
      `--custom-generate-function-path`; be sure to bring the reward de-contamination patch
      `3a6eb2f` along with it
- [ ] Investigate why resuming does not take effect — see "Two operational lessons" below

## End-to-end training: the setup ported over from v1

The goal is to answer one question: does it still learn after migrating to v2? So the
hyperparameters copy v1's 100-step S2TT experiment item by item, and the reward is ported word for
word — shift the convention even slightly and the curves become incomparable. v1's reference
curve: first 10 steps mean BLEU about 0.29, steps 31–40 about 0.41, steps 80–90 about 0.53. The
criterion must use window means, because v1's own single-step swing can be ±0.1 (step 90 is 0.539,
step 100 drops to 0.397).

The first run is fixed at 40 steps: v1 recorded about 4.4 minutes per step, so 40 steps is about 3
hours, which fits inside a 240-minute container window; and steps 30–40 are exactly where the
signal emerges from the noise. Getting to 100 steps would require SAVE_INTERVAL + retries to carry
across containers, and validating the migration does not need to pay that cost.

Three new files:

- `omni_s2tt/bleu_rm.py` — sentence-level BLEU, hooked in via `--custom-rm-path`. v1 modified
  Relax's source to add an `rm_type=bleu` branch; v2 has an extension point, so this delta drops
  from "modify the framework" to "add one file". The algorithm matches v1 word for word,
  deliberately without improvements. Only one observation was added: counting how many responses
  carry `<|...|>` (v1 discovered in the simultaneous work that this contamination pushes BLEU down
  to roughly 40% of its true value), but it does not change the score, because v1's S2TT baseline
  did not have that patch either at the time.
- `omni_s2tt/run-qwen3-omni-lora-s2tt-4gpu.sh` — the v2 version of v1's smoke script.
- `scripts/modal_train_s2tt.py` — the runner, including a CPU-only `check`, resume support, and
  the BLEU-curve verdict.

### Three things that disappeared relative to v1 (all because upstream improved)

1. **The LoRA switch**: v1's `--lora-enable --lora-name policy` does not exist in v2. LoRA is
   turned on by `--lora-rank > 0`, the adapter name is the code constant `LORA_ADAPTER_NAME`, and
   `--lora-adapter-mode` selects adapter mode.
2. **sglang's LoRA arguments**: v1 had to hand-write `--sglang-enable-lora` /
   `--sglang-max-lora-rank` / `--sglang-lora-target-modules`; v2's `sglang_engine.py` derives them
   from the training-side arguments itself (targets go through
   `convert_megatron_to_hf_target_modules` into `q/k/v/o_proj`), and the rollout carries
   `lora_path` automatically. The whole chain is wired up upstream.
3. **A pile of workarounds**: `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`,
   `--sglang-attention-backend triton`, `--sglang-disable-cuda-graph`,
   `--sglang-disable-custom-all-reduce`. These existed in v1 because a newer sglang was injected
   into an old slime image and the kernel binaries did not match; v2 uses the official Relax image
   plus a same-version (0.5.12.post1) sglang fork, so there is no mismatch.

One safe default is kept from v1: **no sequence-parallel, no recompute** (v1 recorded that
recompute + SP + LoRA makes lora_B's backward produce NaN). Note that at TP>1 Omni's provider
enables sequence_parallel itself inside `finalize()`; we do not stack anything on top of that.

`--lora-target-modules` uses the bare names `linear_qkv linear_proj` rather than v1's wildcards:
probe 7 confirmed the towers cannot be wrongly attached (v2's audio/vision are HF modules named
things like `q_proj`, nothing called `linear_qkv`), while wildcards land verbatim in
`adapter_config.json` (filed as Relax #261).

### Data

Reuses v1's `s2tt-data` volume, 128 FLEURS en→zh items, the same batch as v1 used back then, with
no missing audio. Samples look like `{"prompt": "<audio>...", "audios": [wav], "label":
{"ground_truth": ...}, "metadata": {"tgt_lang": "zh", ...}}`. 128 items at rollout-batch 8 is one
epoch every 16 steps, so 40 steps is about 2.5 epochs — v1's 100-step run used the same amount of
data, which is what makes them comparable.

## Probe 9: a three-way comparison of adapter transport (2026-08-12, pure CPU, a few minutes)

On the first attempt at a 40-step training run, the model built, LoRA injected, the engine came
up, and base weight sync completed (19743 parameters, 7.7 seconds) — then it died on the first
adapter push. And only one rank died: TP1–TP3 all printed `loading from tensors completes`, while
TP0 alone reported

```
RuntimeError: unable to open shared memory object </torch_4103_2908601601_198>
in read-write mode: No such file or directory
```

`4103` is the pid of training-side rank 0. **This exposed probe 5's blind spot**: probe 5 validated
in-process hot-loading of `sglang.Engine` and never went through cross-process shared memory at
all, so the conclusion "adapters can be hot-pushed" does not automatically hold over the real
Ray → HTTP path.

An almost free CPU probe (`scripts/probes/modal_probe_transport.py`: one Ray actor serializes, four
consumer actors deserialize) puts the three transport modes side by side:

| Transport | Result |
|---|---|
| `file_descriptor` (torch's default) | All four ranks fail with `AuthenticationError` |
| `file_system` (what upstream actually uses) | TP0 fails with ENOENT, TP1–3 succeed |
| pickle + base64 inlined (what v1 did) | All four ranks pass, checksums all match |

The second row matches the real-hardware error word for word. **The key is that TP0 has to arrive
late for it to reproduce**: that file in `/dev/shm` is reference-counted; the ranks that arrive
first map it and release their reference on return, the count hits zero and the file is unlinked,
and any rank arriving later gets ENOENT when it tries to open it. On the first probe run all four
consumers came in nearly simultaneously and everything passed; adding an 8-second delay reproduced
it immediately. That explains why it was specifically TP0 that died on real hardware, and it also
means this bug has been living on scheduling luck all along.

Worth noting: upstream's comments show they **already hit this once** — the default strategy could
not survive Ray→HTTP, so they deliberately switched to `file_system`. What died on real hardware is
the second pitfall past that workaround. What the two strategies have in common is the actual
problem: the payload carries a reference, and that reference's lifetime depends on whether the
producer is still holding the storage.

The fix is to inline the real bytes on the adapter path, taking the payload from 0.1 MB to 31.6 MB
(a rank-16 adapter is about 24 MB, and v1 carried that cost through 100 steps). The sglang side
needs no change, because `MultiprocessingSerializer.deserialize` is already base64-decode +
unpickle. The base weight path is left alone: those tensors are on the device and serialize to
CUDA IPC handles, which are self-contained — which is also exactly why base weight sync never had
a problem.

## End-to-end 40-step S2TT (2026-08-12, 4×A100-80GB, colocated TP4/EP4)

Ran 40/40 steps, pushing an adapter every step, with no further transport errors. In ten-step
segments against v1 (v1's table is 1-indexed and ours is 0-indexed; already aligned):

| Range | v1 | v2 | Difference |
|---|---|---|---|
| First 10 steps | 0.287 | 0.294 | +0.007 |
| Steps 11–20 | 0.344 | 0.331 | −0.013 |
| Steps 21–30 | 0.345 | 0.357 | +0.012 |
| Steps 31–40 | 0.389 | 0.391 | +0.002 |
| Gain (last segment − first segment) | +0.102 | +0.098 | — |

All four segments land within noise, and the gains are nearly identical. The criterion was fixed
before the run started (last-10 mean landing in 0.36–0.42, with a first-to-last delta of the same
order), not fitted after seeing the result. Per-step values are in
`docs/results/s2tt-40step-curve.json`.

**The precondition for the two curves being comparable** should be stated: v1's 100-step S2TT
baseline is `lora-omni-baseline @ 8bcbb42`, while the reward de-contamination commit `3a6eb2f`
came **afterwards**, for the simultaneous work — which means v1's S2TT curve carries the same
`<|im_end|>` contamination ours does. So not stripping it this time is exactly right. When
comparing simultaneous runs in future, that patch must be ported along, otherwise we replay what
v1 recorded: "BLEU flat at 0.06, advantage ≈ 0, cannot learn".

With that, the v1 → v2 migration closes the loop end to end: sglang's `should_apply_lora` gate,
Omni's `_lora_pattern`, Relax's wildcard expansion and PEFT prefix, and adapter transport were all
implicitly validated by this one training run.

### Two operational lessons

1. **`remote()` binds the training run's lifetime to the local modal process.** On the first
   40-step attempt, the local side dropped and the app was reaped at step 5
   (`Runner has been shutting down for too long`), leaving only `iter_0000004`. Switch to v1's
   approach — `train.spawn(...)`, handing the call to the server side and walking away, paired
   with `modal run --detach`. Fetch results with
   `modal run modal_train_s2tt.py::result --call-id <id>`.
2. **Resuming did not take effect.** On restart the intent was to continue from `iter_0000004`,
   but it actually started from 0, because the previous run had been reaped early and
   `latest_checkpointed_iteration.txt` was never written. This turned out to be a blessing: we got
   a complete 0–39 curve in one piece, which compares against v1 more cleanly. But the LoRA resume
   path is still unverified and needs investigating on its own before any long run.
