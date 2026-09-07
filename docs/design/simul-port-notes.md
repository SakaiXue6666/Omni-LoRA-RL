# Porting simultaneous interpretation: what it would involve

**Status: not ported.** Simultaneous S2TT is not in this tree. It exists on the `v1` branch and
has never been run on the current implementation.

This document is the survey done before attempting the port. It was attempted once and reverted,
so the findings below are analysis, not experience with a working port.

## Where the code is

Everything is on the `v1` branch:

| Path | Lines | What it is |
|---|---|---|
| `Relax/examples/simul_s2tt/rollout.py` | 788 | The body: multi-turn generate, one audio chunk per turn |
| `Relax/examples/simul_s2tt/audio_chunk_env.py` | 103 | 960 ms fixed-chunk env; `step()` ignores the model output and just feeds the next chunk |
| `Relax/examples/simul_s2tt/config.yaml` | 5 | `max_turns: 64` / `simul_chunk_ms: 960` |
| `Relax/examples/simul_s2tt/_selftest_env.py` | 113 | Pure-numpy chunking self-test, no GPU |
| `Relax/examples/simul_s2tt/__init__.py` | 14 | Docstring |
| `Relax/scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-simul.sh` | 168 | **The launch script that produced the recorded result** |
| `Relax/examples/simul_s2tt/omni_rollout.py` | 396 | The sglang-omni Thinker variant — out of scope, the current tree has no sglang-omni submodule |

It works: 20 steps on 2026-07-10, BLEU 0.155 → 0.265
(`docs/results/experiments.md`, experiment 3). The logic is not in question.

**Port the launch script as-is.** It is the artifact that actually produced that number. A
previous attempt replaced it with a thin wrapper around the single-turn script, which required
editing the single-turn script — an already-verified artifact — and was reverted. Copy it, do not
redesign it.

## What the current implementation would require changing

Four things, all in `rollout.py`, all forced by API changes in Relax since v1:

| # | Location | Change | Why |
|---|---|---|---|
| ① | imports + the two `run_in_executor` call sites | `_ENCODE_EXECUTOR` → `get_encode_executor()` | That private global was replaced by a lazily constructed accessor |
| ② | `_run_inference_step` | `post(url, payload)` → `state.post_generate(url, payload)`, taking `state` as a new argument | See "permit" below |
| ③ | end of file | `generate.manages_inference_permit = True` | Pairs with ②; neither works alone |
| ④ | `generate()` | catch `GenerationAborted` | The upstream contract requires it not to escape |

**The permit**: current Relax added `relax/engine/rollout/request_permit.py`, which v1 has no
concept of. A multi-turn rollout that does not declare `manages_inference_permit` holds one slot
of the session-level semaphore for its entire run — all ~10 turns plus the chunking and encoding
time in between. It does not deadlock and does not serialize; it costs throughput. Upstream's own
comment: *"Custom multi-turn rollouts should use inference_permit()/post_generate() instead."*
`Relax/examples/deepeyes/rollout.py` is the reference implementation.

Doing only half of ②③ fails loudly, which is the good case:

```
RuntimeError: inference_permit()/post_generate() was called while the session-level
lock is held. A custom generate function must declare `manages_inference_permit = True`
to use per-request permits; without it ... acquiring a permit would deadlock.
```

> An implementation of all four exists at `12fed1b` on the Relax fork's `simul-port-attempt`
> branch,
> from the reverted attempt. It has never been run. Treat it as a starting point to review, not
> as working code.

---

# The one real trap: the rope device patch

**Read this before running anything.** It is the only known blocker, and it fails in a way that
looks like it is not there.

## What the bug is

`get_rope_index` builds position ids on CPU with `torch.arange(...)`, and the running counters
`st` / `st_idx` accumulate each segment's length into that CPU chain. `audio_seqlens` arrives as a
CUDA tensor — callers derive it from `feature_attention_mask.sum(1)` — so `audio_len` stays on
CUDA and contaminates the counter. On the *next* segment, `st_idx` (CPU) meets `text_len` (now
CUDA) and:

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

The first audio segment is always safe: `st` is still a plain int at that point. **It only fires
once a sequence holds two or more audio segments** — which is why single-turn S2TT never sees it,
and simultaneous interpretation (~10 chunks per sample) hits it immediately.

Inside the same function the video branches already guard exactly this way with
`second_per_grids[video_index].cpu()`; the audio branches do not.

## Why v1 does not hit it, and why copying v1's fix will not work

v1 carries a patch for this — `patch_qwen3_omni_rope_index_device()` in
`Relax/relax/backends/megatron/__init__.py`. It was added in commit `a700c7f`, which is *the same
commit that introduced the simultaneous rollout*: the patch exists because simultaneous
interpretation forced it into being. The recorded 20-step run succeeded because that patch was in
place.

It cannot simply be copied forward, for two reasons:

1. **It patches the wrong module now.** v1 patches
   `megatron.bridge.models.qwen_omni.modelling_qwen3_omni.model.get_rope_index`. Current Relax
   vendors its own copy — `relax/models/qwen_omni/qwen3_omni_provider.py` imports
   `Qwen3OmniMoeModel` from `relax/models/qwen_omni/modeling_qwen3_omni/`, which uses its own
   `utils.get_rope_index`. Different module; the patch never runs.
2. **It fails silently.** The call is wrapped in `try/except ImportError: pass`, so if the target
   module is missing you get no error at all — the patch simply does not apply, and everything
   looks fine until the first training step crashes.

## Status of the fix

**Unresolved.** A two-line fix (`audio_seqlens = audio_seqlens.cpu()` at the top of the vendored
`get_rope_index`) was written and then reverted, because it sits in the forward path of the
already-verified single-turn run and had never been executed on a GPU. It is at `9e94202` on the
fork's `simul-port-attempt` branch if you want to look at it.

Whoever does the port has to decide how to handle this. The options, roughly:

- Apply the fix to the vendored copy and re-verify the single-turn 40-step run still matches
- Keep it out of the shared tree and apply it as a patch only when running simultaneous
- File it upstream — the asymmetry with the video branches makes it a defensible upstream bug
  report, though there is no reproduction on hand and upstream's own workloads look single-audio

Verified at the code level on upstream main `f361a16` (2026-09-04): unchanged, no test coverage,
and no existing issue or PR covers it.

---

# Behavior notes

Not risks — these all worked in v1. They are here because they are unintuitive.

## The dual token stream

`rollout.py` maintains two streams: `sample.rollout_tokens` goes to sglang (one unexpanded marker
per audio segment), `sample.tokens` goes to Megatron (expanded by the processor, aligned with the
audio features), and `_merge_mm_train()` pads each chunk's `input_features` to a common length and
concatenates along dim=0.

When this goes wrong it is usually not an exception but divergent training, because `loss_mask`
and the tokens no longer line up. The invariants v1's CPU self-test asserts:

- `len(loss_mask) == response_length`
- `len(rollout_log_probs) == response_length`
- `len(tokens) == prompt_len + response_length`
- `sum(loss_mask) == total generated tokens` (excluding injected observation tokens)
- number of sglang calls `== num_chunks`
- `input_features` present, first dimension == number of audio segments

`sample.metadata` carries `simul_num_chunks` and `simul_stop_reason`.

## Special-token stripping differs from single-turn

| | `<\|...\|>` special tokens |
|---|---|
| Simultaneous | **Stripped** before concatenating the response (`_clean_gen_text`, commit `3a6eb2f`) |
| Single-turn S2TT | **Not stripped**; warns only, score unchanged |

Simultaneous has to strip: a marker lands between every pair of chunks, destroying every
cross-chunk 2/3/4-gram. Measured, BLEU drops to ~40% of its true value (7.2 vs 17.9). Without it
advantage ≈ 0 and RL cannot learn — that is exactly how v1's first 40-step attempt was wasted.
Single-turn does not strip because every recorded curve was produced that way.

**So the two curves are not directly comparable in absolute value.** Compare each against its own
history.

## `--no-offload-train/rollout`

v1's simultaneous configuration keeps the base model frozen and resident on both sides, which is
~27–35% faster than running with offloading, at the cost of tighter memory
(`sglang-mem-fraction-static` lowered to 0.55).

Both flags currently work — `arguments.py` reads `if args.offload_train is None: = True`, so the
value is only forced when you did not set it explicitly. But the help text in the same place says
*"This will always be true when --colocate is set."* If upstream ever makes the code match the
help, simultaneous silently falls back to offload mode: no error, just slower. If per-step time
jumps from ~3 minutes to ~4.4 minutes, look here.

## Data requirement

`AudioChunkEnv.__init__` asserts exactly one complete audio clip per sample — chunking is the
env's job, so what you feed in must be the whole clip. The 128 items used for single-turn satisfy
this. v1 used 97 full clips averaging 9.6 s (3.8–23.4 s), one chunk per 960 ms → about 10 chunks
per clip.

# Tests available on the v1 branch

Both live in `modal_relax_smoke.py` on `v1`:

- `selftest_simul` (line 2185) → `selftest_simul_encode` — **CPU, about 1 minute.** Builds a
  3-second synthetic clip through the real `process_raw_sample` path (4 chunks, last one
  deliberately short to reproduce variable-length feature concatenation), mocks out
  `GenerateState`/`post`, runs the full multi-turn `generate()`, and asserts the invariants listed
  above. **This is the cheapest way to tell whether changes ①②③④ are correct before spending GPU
  time.**
- `selftest_rope` (line 2313) — T4, 1–2 minutes, reproduces the device bug above against the pure
  function.

`git show v1:modal_relax_smoke.py`, then lift out `_offline_build_simul_out` and
`selftest_simul_encode`.

Note that `_selftest_env.py` is much weaker than it looks: it exercises only the pure-numpy
chunking logic with a fake sample, has no relax dependencies, and `examples/deepeyes/base_env.py`
is byte-for-byte identical between the two versions — so it passes regardless and verifies nothing
about a port.
