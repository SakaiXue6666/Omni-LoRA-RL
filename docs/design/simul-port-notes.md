# Notes on porting simultaneous interpretation

The simultaneous S2TT code sits where it did in v1: `Relax/examples/simul_s2tt/` (inside the
submodule). The launch script `omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh` stays in this repo,
next to the single-turn one.

**The code itself has been run successfully** — 20 steps on the older implementation on
2026-07-10, BLEU 0.155 → 0.265 (`docs/results/experiments.md`, experiment 3). The logic is not in
question.

What has *not* been run is **the port itself**: one patch was lost in the migration, plus the four
edits made here. The two are written up separately below — do not conflate them.

## What was moved

| File | Origin | Notes |
|---|---|---|
| `Relax/examples/simul_s2tt/rollout.py` | v1 file of the same name | The body: multi-turn generate (788 lines) |
| `Relax/examples/simul_s2tt/audio_chunk_env.py` | same | 960 ms fixed-chunk env, logic unchanged |
| `Relax/examples/simul_s2tt/config.yaml` | same | `max_turns: 64` / `simul_chunk_ms: 960` |
| `Relax/examples/simul_s2tt/_selftest_env.py` | same | Pure-numpy chunking self-test |
| `Relax/examples/simul_s2tt/__init__.py` | same | Docstring |
| `omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh` | newly written | Thin wrapper around the single-turn script |

**Not moved:** `omni_rollout.py` (396 lines) — that is the sglang-omni Thinker variant, and the
current code no longer carries the sglang-omni submodule. Grab it from the `v1` branch if needed.

## Why it lives inside the Relax fork

Same as v1, and next to the `examples/deepeyes/base_env.py` it borrows from. The cost is ~1000
lines of extra divergence in the fork, which makes syncing with upstream more conflict-prone; but
any change to Relax has to go through the three-step submodule flow anyway, so the location makes
no difference there. See "Changing Relax / sglang code" in the README.

## The four changes

| # | Location | Change | Why |
|---|---|---|---|
| ① | `rollout.py` imports + lines 638/749 | `_ENCODE_EXECUTOR` → `get_encode_executor()` | Current Relax replaced that private global with a lazily constructed accessor |
| ② | `_run_inference_step` | `post(url, payload)` → `state.post_generate(url, payload)`; the signature takes an extra `state` | See "permit" below |
| ③ | End of file | Added `generate.manages_inference_permit = True` | Pairs with ②; neither works alone |
| ④ | `generate()` | Split into a `generate()` shell plus `_generate_impl()`, with the shell catching `GenerationAborted` | The upstream contract requires it not to escape |

**About the permit:** current Relax added `relax/engine/rollout/request_permit.py` (v1 had no such
thing). A multi-turn rollout that does not declare `manages_inference_permit` holds one slot of
the session-level semaphore for the entire run — across all ~10 turns plus the chunking and
encoding time in between. It does not deadlock and does not serialize; it costs throughput. The
upstream comment puts it exactly: *"Custom multi-turn rollouts should use
inference_permit()/post_generate() instead."*

---

# The two real risks

## 1. The rope device patch (lost in the migration, **now fixed**)

> **Fixed**: commit `9e94202` on the Relax fork's `lora-omni-v2` moves `audio_seqlens` onto the
> CPU directly inside `relax/models/qwen_omni/modeling_qwen3_omni/utils.py`, matching what the
> video branch already does. The background below is kept so it can be sent upstream later.

**This is not a code defect, it is a migration omission.** The evidence is solid:

- The patch `patch_qwen3_omni_rope_index_device()` was added in commit `a700c7f` (2026-07-10),
  whose title is literally "add the Qwen3-Omni simultaneous (multi-turn fixed audio chunk) custom
  rollout" — **the patch and the simultaneous rollout were born in the same commit; the patch
  exists because simultaneous interpretation forced it into being**
- So those 20 steps in v1 succeeded precisely *because* the patch was there. Having run
  successfully is evidence that the bug exists, not evidence that it does not
- Current Relax does not carry the patch, and the equivalent code at its new location is
  unchanged

**Why v1's patch cannot simply be copied over:** it patches
`megatron.bridge.models.qwen_omni.modelling_qwen3_omni.model.get_rope_index`, whereas
`relax/models/qwen_omni/qwen3_omni_provider.py:23` imports Relax's **own vendored**
`Qwen3OmniMoeModel`, which goes through its own `modeling_qwen3_omni/utils.py`. Two different
modules. Worse, v1's code is wrapped in `try/except ImportError: pass`, so **it fails silently
when it fails to apply**.

Verified link by link at the code level, never reproduced on real hardware:

```
model.py:206   audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)   # CUDA
model.py:368   get_rope_index(..., audio_seqlens=audio_feature_lengths)           # no .cpu()
utils.py:9     _get_feat_extract_output_lengths()  pure arithmetic, device unchanged  # still CUDA
utils.py:188   audio_len = _get_feat_extract_output_lengths(audio_seqlens[i])     # CUDA scalar
utils.py:192   st += text_len + bos_len + audio_len + eos_len                     # poisons the CPU counter
```

In the same function the **video branches call `.cpu()` explicitly** (`utils.py:235`, `271`); the
audio branch does not — the same shape of bug v1 hit back then.

**It only fires once a sequence holds two or more audio segments.** Single-turn S2TT has exactly
one per item and never reaches it (which is why the 40-step run went through cleanly);
simultaneous interpretation has ~10 chunks per item and hits it every time.

Expected error:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

`get_rope_index` will be in the traceback. **The fix** (copy what the video branch does — it
changes the compute device only, not the values): at the call site `model.py:368` pass
`audio_seqlens=audio_feature_lengths.cpu()`, or call `.cpu()` in `utils.py` before entering the
audio branch.

This one could also go straight upstream as a sixth PR — same character as the five already
filed: upstream's own code guards the video branch with `.cpu()` and forgot the audio branch.

## 2. Changes ②③ getting out of sync (permit)

Changes ② and ③ are a pair. Doing only half of it raises immediately:

```
RuntimeError: inference_permit()/post_generate() was called while the session-level
lock is held. A custom generate function must declare `manages_inference_permit = True`
to use per-request permits; without it ... acquiring a permit would deadlock.
```

The good news is it **fails loudly rather than deadlocking silently**. If you see this, check
whether that line at the end of the file is still there.

---

# The following are not risks, they are behavior notes

These four were originally listed as "risks" as well, which was an overreading — v1 ran fine, so
they all work as intended. They are kept here because they are unintuitive and easy to misjudge
on first contact.

## 3. The dual token stream alignment (verified correct in v1)

`rollout.py` maintains two streams: `sample.rollout_tokens` goes to sglang (one unexpanded marker
per audio segment), `sample.tokens` goes to Megatron (expanded by the processor, aligned with the
audio features), and `_merge_mm_train()` pads each chunk's `input_features` to a common length and
concatenates along dim=0.

This logic was verified correct in v1 and not a single line of it changed in the port. The only
uncertainty is whether transformers/processor behavior differs now that the image changed — I have
no specific evidence that it did, it simply has not been run.

When this does go wrong it usually is not an exception but **divergent training or an odd loss**,
because `loss_mask` and the tokens no longer line up. The invariants to check (v1's CPU self-test
asserted exactly these):

- `len(loss_mask) == response_length`
- `len(rollout_log_probs) == response_length`
- `len(tokens) == prompt_len + response_length`
- `sum(loss_mask) == total generated tokens` (excluding the injected observation tokens)
- number of sglang calls `== num_chunks`
- `input_features` present, with its first dimension == number of audio segments

`sample.metadata` carries `simul_num_chunks` and `simul_stop_reason` to check against.

## 4. `--no-offload-train/rollout` works today, but the help text disagrees with the code

The launch script uses both flags (v1's resident configuration, ~27–35% faster than running with
offloading). They currently **do work** — `arguments.py:3170` reads
`if args.offload_train is None: = True`, so it is only forced when you did not set it explicitly.

But the help text in the same place says *"This will always be true when --colocate is set."* The
two disagree. The day upstream changes the code to match the help, simultaneous interpretation
will **silently fall back to offload mode** — no error, just slower with different memory
behavior. If per-step time suddenly jumps from ~3 minutes to ~4.4 minutes, look here first.

## 5. Special-token stripping differs from single-turn (by design)

| | `<\|...\|>` special tokens |
|---|---|
| Simultaneous (this module) | **Stripped** before concatenating the response (`_clean_gen_text`, `rollout.py:156`) |
| Single-turn S2TT | **Not stripped**; warns only, score unchanged |

This is not a bug; each has its own history:

- Simultaneous *has* to strip — a marker gets inserted between every pair of chunks, which
  destroys every cross-chunk 2/3/4-gram. Measured, BLEU is pushed down to ~40% of its true value
  (7.2 vs 17.9). Without the fix advantage ≈ 0 and RL simply cannot learn (that is exactly how
  v1's first 40-step run was wasted).
- Single-turn does not strip — every recorded curve was produced under contaminated conditions,
  and changing it would make them incomparable.

**Consequence: absolute values from the simultaneous curve and the single-turn curve are not
directly comparable.** Compare each against its own history.

## 6. Data requirement: exactly one full audio clip per sample (same assert as v1)

`AudioChunkEnv.__init__` has a hard assertion:

```python
assert len(audios) == 1, "AudioChunkEnv expects exactly 1 complete audio clip initially..."
```

Chunking is done by the env itself, so what you feed in must be the **complete** clip. The 128
items used for single-turn satisfy this. v1 used 97 full clips averaging 9.6 seconds (3.8–23.4 s),
one chunk per 960 ms → roughly 10 chunks per clip.

---

## Tests that were not brought over

v1 had two self-tests, both still living in `modal_relax_smoke.py` on the `v1` branch; neither was
ported:

- `selftest_simul` (line 2185) → `selftest_simul_encode` — **CPU, about 1 minute**. Builds a
  3-second synthetic clip through the real `process_raw_sample` path (4 chunks, with a
  deliberately short final chunk to reproduce variable-length feature concatenation), mocks out
  `GenerateState`/`post`, runs the full multi-turn `generate()`, and asserts the invariants listed
  in section 3 above. **This is the only thing that can tell you whether changes ①②③ are correct
  before you spend GPU time.**
- `selftest_rope` (line 2313) — T4, 1–2 minutes, reproduces the device bug from section 1 against
  the pure function.

To use them: `git show v1:modal_relax_smoke.py`, then lift `_offline_build_simul_out` and
`selftest_simul_encode` out and adapt them.

The `_selftest_env.py` that *was* brought over is of limited use: it exercises only the pure-numpy
chunking logic with a fake sample built from `SimpleNamespace`, has zero relax dependencies, and
`examples/deepeyes/base_env.py` is byte-for-byte identical across the two versions — so it is
guaranteed to pass and verifies nothing related to this port. It is kept because it is still
useful when changing the chunking logic.

## How to run it

```bash
HF_CKPT=/models/qwen3-omni DATA=/s2tt/train_s2tt.jsonl NUM_ROLLOUT=20 \
SAVE_DIR=/data/s2tt/ckpt/simul_run1 \
bash omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh
```

It is a thin wrapper around the single-turn script: all hyperparameters are reused, and it only
adds `--custom-generate-function-path` / `--custom-config-path` plus the resident configuration.
The log file and the tensorboard project name are already separated from the single-turn ones, so
they will not get mixed together.
