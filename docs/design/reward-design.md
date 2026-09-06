# Why the reward is sentence-level BLEU

Reading the code you will run into a few constants with no comment attached —
`ROLLOUT_TEMPERATURE=1.1`, `N_SAMPLES=8`, and the choice of BLEU over accuracy. The reasons are
all here. They came out of experiments, not out of thin air.

## What GRPO needs is not a high score, it is variance within the group

GRPO's advantage comes from the *relative* differences among the responses sampled for the same
prompt. If every sample in a group gets the same score, the advantage is 0 and so is the
gradient — even when that score is a perfect one.

This is not a theoretical worry. We ran it once with hard multiple-choice math: the loop was
fine, the reward function was correct, and yet **`raw_reward = 1.0`, 32/32 all correct**.
Qwen3-Omni-30B's thinker with CoT nails 2–3 digit multiplication every time. Zero variance,
nothing to learn from.

We tried making it harder so it would occasionally get one wrong: thinking on = 100% correct;
thinking off, or `max_response_len` cut to 96 tokens = 0% (a fake all-wrong, because `<answer>`
gets truncated). **There is no middle ground.** So "just make the task harder" does not work on
this model.

**Conclusion: a 0/1 binary reward is unusable at this model scale.**

## Hence a continuous reward

Sentence-level BLEU is in [0, 1], naturally continuous, and the model almost never scores a
perfect 1 — there is always headroom, so there is always variance within the group. The first
10-step experiment after the switch already showed it rising (0.463 → 0.523).

## Three levers for creating in-group variance

Picking a continuous reward is not enough. Variance has to be manufactured deliberately. All
three levers are in use:

1. **The task itself must admit several valid answers.** Translation does this naturally: the
   same sentence has multiple correct renderings, and single-reference BLEU lands in the 0.3–0.6
   range where there is something to discriminate. The early validation deliberately used long,
   difficult sentences (subordinate clauses, colloquial speech) to spread the distribution out.
2. **Temperature.** The sampling temperature decides how spread out the 8 responses to one prompt
   are. Too low → all 8 nearly identical → back to zero variance. S2TT currently uses **1.1**;
   the early text-only translation validation used 1.3.
3. **Enough data to average out batch noise.** In-group variance is wanted; batch-to-batch noise
   is not. With too small a dataset, the difficulty swing between the prompts drawn each step
   drowns out the learning signal.

`ROLLOUT_TEMPERATURE=1.1` and `N_SAMPLES=8` are lever 2 and "how many per group".
**Before lowering either the temperature or n-samples, think about whether the group still has
any variance left.**

## A recurring pitfall: the reward gets contaminated by special tokens

The text sglang returns sometimes carries special tokens such as `<|im_end|>`. Concatenating
those into the response before computing BLEU pushes the score down to roughly 40% of its true
value. The first simultaneous-interpretation run had BLEU flat at 0.06 with advantage ≈ 0 for
exactly this reason — not because the model could not do the task.

`omni_s2tt/bleu_rm.py` detects this and prints `[bleu_rm] special token found in response`, but
it **only reports, it does not change the score**. A few stray occurrences are fine; if they show
up in bulk it means something is wrong on the rollout side. Go look there — do not tune the
reward.

## Related

- Full data for every experiment: `docs/results/experiments.md`
- The reward implementation: `omni_s2tt/bleu_rm.py` (sacreBLEU with the Chinese tokenizer,
  divided by 100 to normalize into [0, 1])
