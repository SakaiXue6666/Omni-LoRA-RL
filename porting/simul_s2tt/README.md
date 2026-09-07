# Simultaneous interpretation — v1 code, not ported

Everything here is copied **byte-for-byte** from the `v1` branch. Nothing in this folder is wired
into the current tree: no import path reaches it, no script runs it, and the training code does
not know it exists. It is parked here so the port does not have to start by archaeology.

It works — on the older implementation. 20 steps on 2026-07-10, BLEU 0.155 → 0.265
(`docs/results/experiments.md`, experiment 3).

What porting it to the current implementation involves is written up in
[`docs/design/simul-port-notes.md`](../../docs/design/simul-port-notes.md). **Read that before
touching anything here.**

## What is in this folder

| Path | Origin on `v1` | Notes |
|---|---|---|
| `src/rollout.py` | `Relax/examples/simul_s2tt/rollout.py` | The body: multi-turn generate, one audio chunk per turn (788 lines) |
| `src/audio_chunk_env.py` | same directory | 960 ms fixed-chunk env; `step()` ignores the model output and feeds the next chunk |
| `src/config.yaml` | same directory | `max_turns: 64` / `simul_chunk_ms: 960` |
| `src/_selftest_env.py` | same directory | Pure-numpy chunking self-test, no GPU |
| `src/__init__.py` | same directory | Docstring |
| `run-qwen3-30B-A3B-omni-lora-simul.sh` | `Relax/scripts/training/multimodal/` | **The launch script that produced the recorded result** |
| `main-code-rope-patch.diff` | commit `a700c7f` | The only main-code change simultaneous interpretation needs — see below |

Deliberately **not** copied: `omni_rollout.py` (396 lines), the sglang-omni Thinker variant. The
current tree has no sglang-omni submodule. It is on `v1` if it is ever needed.

## The one main-code change

Simultaneous interpretation requires exactly one change outside `examples/simul_s2tt/`:
`patch_qwen3_omni_rope_index_device()` in `relax/backends/megatron/__init__.py`, +40 lines,
captured in `main-code-rope-patch.diff`.

We know it is required rather than incidental because it was added **in the same commit that
introduced the simultaneous rollout** (`a700c7f`) — the patch exists because simultaneous
interpretation forced it into being. The recorded 20-step run succeeded because it was in place.

**The bug it works around**: `get_rope_index` builds position ids on CPU, but `audio_seqlens`
arrives as a CUDA tensor, so the running counter is contaminated and the *second* audio segment
raises `Expected all tensors to be on the same device, cuda:0 and cpu`. The first segment is
always safe, which is why single-turn S2TT never sees it and simultaneous interpretation (~10
chunks per sample) hits it immediately.

### ⚠️ This diff does not apply cleanly to the current tree, and it fails silently

Two separate problems, both covered in `docs/design/simul-port-notes.md`:

1. **Wrong module.** The patch targets
   `megatron.bridge.models.qwen_omni.modelling_qwen3_omni.model.get_rope_index`. Current Relax
   vendors its own copy: `relax/models/qwen_omni/qwen3_omni_provider.py` imports
   `Qwen3OmniMoeModel` from `relax/models/qwen_omni/modeling_qwen3_omni/`, which uses its own
   `utils.get_rope_index`. The patch would run against a module nothing calls.
2. **It swallows its own failure.** The call site is wrapped in `try/except ImportError: pass`, so
   if the target module is absent you get no error at all — it looks applied when it is not, and
   the first training step crashes with no clue why.

So treat the diff as **documentation of what needs to happen**, not as something to `git apply`.

An attempted two-line equivalent against the vendored copy exists at `9e94202` on the Relax fork's
`simul-port-attempt` branch. It has never been run on a GPU and was kept out of the shared tree
because it sits in the forward path of the already-verified single-turn run.

## Two things worth knowing before you start

**Copy the launch script as-is.** It is the artifact that produced the recorded number. A previous
attempt replaced it with a thin wrapper around the single-turn script, which required editing the
single-turn script — an already-verified artifact — and was reverted.

**Run the cheap test first.** `v1`'s `modal_relax_smoke.py` has `selftest_simul` (line 2185) →
`selftest_simul_encode`: CPU, about a minute, runs the full multi-turn `generate()` against a
synthetic clip with mocked networking and asserts the token/loss-mask invariants. It is the only
thing that catches an incorrect port before you spend GPU time.
`git show v1:modal_relax_smoke.py`, then lift out `_offline_build_simul_out` and
`selftest_simul_encode`.

(`src/_selftest_env.py` is much weaker than it looks — pure numpy, no relax dependencies, passes
regardless of whether a port is correct.)
