# Experiment log

Every RL experiment this project has run, in chronological order. The per-step raw data lives in
the `*.json` files in this directory; each one carries a `_meta` block stating the metric
convention, hardware, data and code path.

One convention throughout: **the metric is `rollout/raw_reward` per step, i.e. the mean
sentence-level BLEU over that batch of samples**; the criterion is always a **ten-step window
mean**, never a single step — single-step noise is large, and on this pipeline we have seen it
drop from 0.539 to 0.397 in one step.

| # | Experiment | Data | Steps | Result (first window → last window) | Raw data |
|---|---|---|---|---|---|
| 0 | Math MCQ + 0/1 reward | hard multiple-choice math | — | **Failed**, zero variance | — |
| 1 | zh→en translation + sentence BLEU | 256 hand-made long sentences | 10 | 0.463 → 0.523 | `translate-10step-curve.json` |
| 2 | **en→zh S2TT (offline, single-turn)** | FLEURS, 128 items | **100** | **0.287 → 0.487** | `s2tt-100step-curve.json` |
| 3 | Simultaneous (960 ms fixed chunks, multi-turn) | FLEURS, 97 full clips | 20 | 0.155 → 0.265 | `simul-20step-curve.json` |
| 4 | en→zh S2TT (current code path) | FLEURS, 128 items | 40 | 0.294 → 0.391 | `s2tt-40step-curve.json` |

---

## 0. Math MCQ: a useful failure (early June 2026)

We ran colocated RL on hard multiple-choice math. The loop was fine and the reward function was
correct (1.0 for a right answer), but **`raw_reward = 1.0`, 32/32 all correct**. Qwen3-Omni-30B's
thinker with CoT nails 2–3 digit multiplication, so in-group variance was zero → advantage = 0 →
no gradient.

We tried to lower the difficulty: thinking on = 100%; thinking off, or `max_response_len` cut to
96 tokens = 0% (a fake all-wrong, because `<answer>` gets truncated). **There is no transition
band in between**, so cutting tokens was not a viable route.

This conclusion directly determined the reward design of every experiment that followed — see
`docs/design/reward-design.md`.

## 1. zh→en translation + sentence-level BLEU (2026-06-02, 10 steps)

A feasibility check after switching to a continuous reward. The per-step curve:

```
0.48  0.44  0.47  0.52  0.47  0.55  0.53  0.50  0.50  0.57
```

First 3 steps mean 0.463 → last 3 steps mean 0.523, Δ = +0.060, verdict "learning signal present
and trending up".

> The original record's summary says 0.463 → 0.525 / Δ +0.061. The per-step values were only kept
> to two decimals at the time; computing from those ten numbers gives 0.523 / +0.060. The
> difference is rounding and does not affect the conclusion.

Around the same time (2026-06-03) we also ran a causal check on the audio path: pairing prompt A
with audio B made the output switch to B's content immediately, and BLEU against reference A
collapsed from 0.435 to **0.004**; all 4 cross combinations across 2 pairs behaved this way. That
proves the audio really is encoded and driving generation, rather than a language prior answering
from memory.

## 2. en→zh S2TT, offline single-turn, 100 steps (started 2026-06-04, continued to 100 on 07-08)

**This is the most complete curve in the project.**

Configuration: 4×A100-80GB colocated, TP4 / EP4 / PP1; LoRA rank 16 / alpha 32 attached to the
thinker language model's `qkv_proj` + `o_proj`; GRPO with kl-loss-coef 0; adam lr 1e-4 constant;
rollout-batch 8 / n-samples 8 / global-batch 64 / temperature 1.1; reward sacreBLEU (Chinese
tokenizer) / 100.

The data is the first 128 items of FLEURS validation — the count was not written into the log at
the time; this number comes from `prep_s2tt`'s default `limit=128`, and it is the same data on
the same volume that experiment 4 used. 128 items at rollout-batch 8 is one epoch every 16 steps,
so 100 steps is about 6.25 epochs.

Ten-step window means:

| Range | BLEU | | Range | BLEU |
|---|---|---|---|---|
| 1–10 | 0.2868 | | 51–60 | 0.4137 |
| 11–20 | 0.3442 | | 61–70 | 0.4400 |
| 21–30 | 0.3448 | | 71–80 | 0.4822 |
| 31–40 | 0.3886 | | 81–90 | 0.4890 |
| 41–50 | 0.4020 | | 91–100 | 0.4873 |

**First 10 steps 0.287 → last 10 steps 0.487, Δ = +0.201.** Range [0.234 @step3, 0.610 @step95].
All ten windows rise monotonically except the last, which is flat.

Supporting evidence: response length dropped from 28 tokens to 20 (tighter translations), and
log_probs rose (more confident generation). A sampled translation, "东非岛屿位于非洲东海岸外的
印度洋上", scored BLEU 0.659. Zero errors across all 100 steps.

This curve was produced in two sittings: steps 1–40 on 06-04, then continued from the checkpoint
to step 100 on 07-08. The first 40 values are identical digit for digit across both records — it
is one run.

## 3. Simultaneous interpretation: 960 ms fixed-chunk multi-turn rollout (2026-07-10, 20 steps)

The full audio clip is split into 960 ms chunks inside the env; each turn feeds one chunk and the
model emits an incremental translation. The reward is BLEU over the concatenation of the whole
translation.

Data: FLEURS en→zh, 97 full audio clips averaging 9.6 seconds (3.8–23.4 s), one chunk per 960 ms
→ roughly 10 chunks per clip. Differences from the offline baseline:
`--custom-generate-function-path` points at `examples.simul_s2tt.rollout.generate`, with
`max_turns=64` and `simul_chunk_ms=960`; the base model is frozen and resident on both sides
(`--no-offload-train --no-offload-rollout`), and `sglang-mem-fraction-static` is lowered to 0.55
to make room for it. Cost is ~2.8–3.2 min/step (about 27–35% faster than the ~4.4 min/step with
offloading).

**First third (steps 1–7) 0.155 → last third (steps 14–20) 0.265, Δ = +0.110.** Range
[0.064 @step2, 0.339 @step18].

**It learns, but the absolute value is only about half the offline number. The reason is
diagnosed and it is not a bug:**

1. **No read/write supervision or penalty.** There is no signal for "wait when you should wait",
   so the model either forces out a fragment every chunk (choppy, repetitive, which drags BLEU
   down) or gives up.
2. **Fragmented chunks lose context.** 960 ms is roughly 1–2 English words; translating each
   chunk independently destroys sentence-level context.
3. **There is no latency penalty in the reward**, so "being simultaneous" is not actually being
   optimized for.

Real simultaneous interpretation needs some or all of: semantic-unit chunking, a cold-start SFT
phase, and a latency-aware reward. What this run validated is that **the multi-turn fixed-chunk
path works end to end and the reward goes up** — not the quality of the interpretation itself.

> One pitfall: an early simultaneous run had BLEU flat at 0.06 with advantage ≈ 0, because the
> reward was contaminated by special tokens such as `<|im_end|>`. The curve above is from a fresh
> 20-step run started from scratch after that fix (`3a6eb2f`).

## 4. en→zh S2TT, current code path, 40 steps (2026-08-12)

A rerun after migrating to the implementation now on main, answering one question: does it still
learn under the new implementation? Hyperparameters match experiment 2 item by item (LoRA 16/32,
TP4/EP4, temperature 1.1, rollout-batch 8 / n-samples 8 / global-batch 64, lr 1e-4), data is the
first 128 items of FLEURS validation.

| Range | BLEU |
|---|---|
| Steps 1–10 | 0.2937 |
| Steps 11–20 | 0.3308 |
| Steps 21–30 | 0.3571 |
| Steps 31–40 | 0.3914 |

**0.294 → 0.391, Δ = +0.098.** Segment by segment this matches the same span of experiment 2
(0.287 / 0.344 / 0.345 / 0.389) to within ±0.013 everywhere.

The criterion was fixed before the run started (last-10 mean landing in 0.36–0.42, with a
first-to-last delta of the same order), not fitted after seeing the result.

**Limit: the current code path has only been verified to 40 steps.** The 100-step curve in
experiment 2 was produced on the older implementation. Whether it keeps climbing to 0.487 past
step 40 has not been verified on the current code.

---

## A known discrepancy in convention

`omni_s2tt/curve.py` computes from the **per-sample rewards** in `rollout_result/train/*.jsonl`,
whereas every number above comes from **`rollout/raw_reward` in the training log**. The two should
be equal (the latter is the batch mean of the former), but they have never been cross-checked. If
your own numbers differ slightly from these, rule out the convention mismatch before suspecting
the training.

## Not yet verified

- Resuming: on 2026-08-12 we meant to continue from `iter_0000004` but it actually restarted from
  0 — the previous run was killed early and `latest_checkpointed_iteration.txt` was never written.
  The LoRA resume path has never been verified on its own.
- Running the current code path for the full 100 steps.
- Simultaneous interpretation on the current code path — the code has been ported
  (`Relax/examples/simul_s2tt/`) but never run.
