# Background
- slime
    - relax: supports audio input
    - miles: supports lora
- verl
- ms-swift

# Done
- Qwen3-Omni thinker + LoRA (qkv, o) doing offline speech-to-text translation (English audio →
  Chinese); the reward (sacreBLEU quality) rises with training within 100 steps.

## Parameter snapshot (rollback baseline, frozen before touching 2a/2b)
> Rollback point: Relax branch `lora-omni-baseline` @ `8bcbb42` (the smoke script includes the
> resume fix); Modal entry point `modal_relax_smoke.py::learn_audio` on the `v1` branch (task
> `s2tt`). Below are the effective parameters after combining the two.
>
> - Model / deployment: Qwen3-Omni-30B-A3B thinker, bf16; colocated, single node 4×A100-80GB;
>   offload on (the colocate default).
> - Parallelism: TP=4 / EP=4 / ETP=1 / PP=1 / CP=1; micro-batch=1.
> - LoRA: rank=16, alpha=32, dropout=0.0, name=policy; target = the thinker language model's
>   qkv_proj / o_proj.
> - GRPO: advantage=grpo, kl-loss-coef=0, entropy-coef=0, eps-clip=0.2 / eps-clip-high=0.28;
>   max-staleness=0.
> - Optimizer: adam, lr=1e-4, lr-decay-style=constant, weight-decay=0, beta=(0.9, 0.95),
>   optimizer-cpu-offload, precision-aware-optimizer.
> - Rollout: rm-type=bleu, num-rollout=100, rollout-batch=8, n-samples-per-prompt=8,
>   global-batch=64, temperature=1.1, max-response=512, max-prompt=4096,
>   multimodal-keys={"audio":"audios"}.
> - sglang: rollout-num-gpus-per-engine=4, mem-fraction-static=0.7, enable-lora, max-lora-rank=16,
>   max-loras-per-batch=1, lora-target-modules=qkv_proj o_proj, attention-backend=triton,
>   disable-cuda-graph, disable-custom-all-reduce.
> - Megatron misc: attention-backend=flash, no-rope-fusion, dropout=0, grad-allreduce-fp32,
>   attention-softmax-fp32.
> - Storage: save=load=/s2tt/ckpt/s2tt_probe, save-interval=5, max-actor-ckpt-to-keep=1,
>   override-opt_param-scheduler.
> - Modal: 4×A100-80GB, retries=10, --detach.

> Per-step raw_reward (BLEU, batch mean):
>
> | Step | BLEU | Step | BLEU | Step | BLEU | Step | BLEU |
> |----|------|----|------|----|------|----|------|
> | 1 | 0.257 | 11 | 0.352 | 21 | 0.318 | 31 | 0.412 |
> | 2 | 0.276 | 12 | 0.324 | 22 | 0.373 | 32 | 0.402 |
> | 3 | 0.234 | 13 | 0.362 | 23 | 0.320 | 33 | 0.397 |
> | 4 | 0.376 | 14 | 0.355 | 24 | 0.358 | 34 | 0.325 |
> | 5 | 0.267 | 15 | 0.282 | 25 | 0.376 | 35 | 0.391 |
> | 6 | 0.260 | 16 | 0.400 | 26 | 0.366 | 36 | 0.278 |
> | 7 | 0.251 | 17 | 0.363 | 27 | 0.346 | 37 | 0.431 |
> | 8 | 0.321 | 18 | 0.328 | 28 | 0.343 | 38 | 0.414 |
> | 9 | 0.322 | 19 | 0.347 | 29 | 0.342 | 39 | 0.418 |
> | 10 | 0.304 | 20 | 0.329 | 30 | 0.306 | 40 | 0.418 |

> | Step | BLEU | Step | BLEU | Step | BLEU |
> |----|------|----|------|----|------|
> | 41 | 0.407 | 51 | 0.333 | 61 | 0.365 |
> | 42 | 0.416 | 52 | 0.402 | 62 | 0.482 |
> | 43 | 0.416 | 53 | 0.457 | 63 | 0.453 |
> | 44 | 0.396 | 54 | 0.446 | 64 | 0.472 |
> | 45 | 0.378 | 55 | 0.430 | 65 | 0.504 |
> | 46 | 0.358 | 56 | 0.390 | 66 | 0.505 |
> | 47 | 0.458 | 57 | 0.419 | 67 | 0.408 |
> | 48 | 0.361 | 58 | 0.391 | 68 | 0.364 |
> | 49 | 0.386 | 59 | 0.445 | 69 | 0.432 |
> | 50 | 0.444 | 60 | 0.424 | 70 | 0.415 |

> | Step | BLEU | Step | BLEU | Step | BLEU |
> |----|------|----|------|----|------|
> | 71 | 0.371 | 81 | 0.475 | 91 | 0.425 |
> | 72 | 0.558 | 82 | 0.517 | 92 | 0.456 |
> | 73 | 0.478 | 83 | 0.459 | 93 | 0.469 |
> | 74 | 0.506 | 84 | 0.492 | 94 | 0.497 |
> | 75 | 0.571 | 85 | 0.511 | 95 | 0.610 |
> | 76 | 0.504 | 86 | 0.565 | 96 | 0.467 |
> | 77 | 0.403 | 87 | 0.484 | 97 | 0.554 |
> | 78 | 0.484 | 88 | 0.462 | 98 | 0.470 |
> | 79 | 0.416 | 89 | 0.386 | 99 | 0.528 |
> | 80 | 0.531 | 90 | 0.539 | 100 | 0.397 |

- Simultaneous interpretation (multi-turn fixed-length audio-chunk rollout; the full clip is split
  into 960 ms chunks inside the env, each turn feeds one chunk and produces an incremental
  translation, and the reward is BLEU over the concatenated whole)

## Parameter snapshot (differences from the s2tt baseline above; everything else is the same)
> Relax branch `lora-omni-baseline`: `3a6eb2f` (reward-fix) + `bb7642f` (no-offload);
> Modal entry point `modal_relax_smoke.py::learn_simul --tag v2` on the `v1` branch
> (save/load=/s2tt/ckpt/s2tt_probe_simul_v2, trained from scratch).
>
> - Simultaneous-specific: `--custom-generate-function-path examples.simul_s2tt.rollout.generate`
>   plus `--custom-config-path examples/simul_s2tt/config.yaml` (`max_turns=64`,
>   `simul_chunk_ms=960`).
> - Data: FLEURS en→zh, 97 full audio clips on the volume, averaging 9.6 s (3.8–23.4 s), 960 ms
>   per chunk → about 10 chunks per clip on average (4–25).
> - Deployment differences (the 2a resident optimization, now folded into the simul script):
>   `--no-offload-train --no-offload-rollout`, `--sglang-mem-fraction-static 0.55` (to make room
>   for the resident base). Base frozen and resident on both sides.
> - Everything else (TP4/EP4, LoRA r16/α32 on qkv_proj+o_proj, GRPO kl=0, adam lr=1e-4 constant,
>   rm=bleu, rollout-batch=8, n-samples=8, global-batch=64, temp=1.1, max-resp=512,
>   max-prompt=4096) matches the s2tt baseline.
> - Cost: ~2.8–3.2 min/step (about 27–35% faster than the ~4.4 min/step with offloading), no OOM.

## Two key fixes
> 1. **Reward de-contamination** (`3a6eb2f`): the text sglang returns each turn carries
>    `<|im_end|>`, and it used to be concatenated straight into `sample.response` and fed to BLEU,
>    which both adds tokens the reference does not have and breaks every cross-chunk n-gram.
>    Measured (Chinese tokenizer), BLEU was pushed down to ~40% of its true value (7.2 vs 17.9,
>    ≈2.5x) and in-group variance was flattened → advantage ≈ 0, RL could not learn. Fix: strip
>    `<|...|>` with a regex before concatenating.
> 2. **no-offload speedup** (`bb7642f`): the 2a resident-base optimization had only been applied to
>    the noffload script and was missed in the simul script, so every step still went
>    offload → wake_up base with no improvement in cost. Both flags plus mem-fraction 0.55 have
>    now been folded into the simul script.

> Per-step raw_reward (batch-mean BLEU, a fresh 20 steps, after the reward fix):
> Note: step K = rollout dump number (K-1) (the framework is 0-indexed).
>
> | Step | BLEU | Step | BLEU | Step | BLEU | Step | BLEU |
> |----|------|----|------|----|------|----|------|
> | 1 | 0.110 | 6 | 0.211 | 11 | 0.163 | 16 | 0.233 |
> | 2 | 0.064 | 7 | 0.199 | 12 | 0.262 | 17 | 0.264 |
> | 3 | 0.137 | 8 | 0.215 | 13 | 0.205 | 18 | 0.339 |
> | 4 | 0.204 | 9 | 0.220 | 14 | 0.303 | 19 | 0.234 |
> | 5 | 0.158 | 10 | 0.240 | 15 | 0.249 | 20 | 0.231 |
>
> Trend: first third (steps 1–7) mean ≈ 0.15 → last third (steps 14–20) mean ≈ 0.27, a clear rise
> (min 0.064 @step2, max 0.339 @step18).
> Conclusion: after the reward fix RL really does learn (while contaminated it sat flat at 0.06
> with advantage ≈ 0); but this is a translation-quality gain under "whole-clip BLEU", not true
> simultaneous interpretation (no read/write supervision, fragments encoded acoustically in
> isolation, no latency penalty in the reward) — real simultaneous interpretation needs
> semantic-unit chunking / cold-start SFT / a latency-aware reward.


# TODO
- 100 steps ✅
- Weight-sync optimization (current state: the full base is pushed every step, and LoRA is
  unloaded→reloaded every step — two wastes): freeze the base, stop offloading/onloading, keep
  both sides resident ✅
- Switch to simultaneous interpretation, i.e. multi-turn with one chunk fed per turn, and see
  whether the reward rises ✅
- Add mlp / moe to LoRA
- Switch to speech-to-speech (training the thinker)
- Switch to fully async

# Other approaches
- ms-swift
- verl

(Adding mlp / moe to LoRA)
My judgement above may have been wrong. The filenames alone are strong counter-evidence: sglang
has lora_moe_runners.py, triton_ops/fused_moe_lora_kernel.py, triton_ops/virtual_experts.py, and
the tests even include test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py (exactly your model)
and test_lora_moe_tp_logprob_diff.py. That flatly contradicts my claim that "sglang does not
support MoE LoRA". Let me read the actual source.

On the Megatron side the matcher's default targets already include linear_fc1/linear_fc2, and it
matches on module name plus type. The key question is how it handles MoE grouped experts. Next,
read Megatron-Bridge's lora.py, plus sglang's lora layers.py and that 30B-A3B MoE test.

This is decisive evidence, and most of my earlier judgement was wrong. Looking at
Megatron-Bridge's LoRA:

MoE experts are explicitly supported: there are is_expert_linear and is_expert flags,
normalize_moe_lora (rank normalization specifically for MoE), and even an
`else: # TE Grouped Linear` branch inside LoRAMerge.transform that iterates
`for i in range(module.to_wrap.num_gemms)` gemm by gemm (lines 351–361).
It also supports attaching LoRA to the router (LoRATopKRouter).
So "can Megatron attach to grouped experts" is not an open question at all — it is natively
supported. Let me now confirm sglang's MoE LoRA support and the naming format it expects, by
reading that 30B-A3B test and the lora layers.

Checked, and I have to retract my earlier judgement — those two "dealbreakers" were basically
wrong. Here are the facts:

Correction: both ends natively support MoE LoRA
① sglang side (I said "almost certainly unsupported" → wrong)

There is a dedicated FusedMoEWithLoRA class (layers.py:859), lora_moe_runners.py, and
fused_moe_lora_kernel.py.
There is a LoRA regression test targeting Qwen3-30B-A3B-Instruct-2507 (exactly your thinker's text
backbone), and what it validates is LoRA logprob alignment with the trainer (KL < 5e-3);
enable_lora=True plus the MoE runner works out of the box.
The engine parameters include MoE-LoRA-specific switches such as moe_runner_backend and
experts_shared_outer_loras=True.
② Megatron-Bridge side (I said "whether it can attach is a big question mark" → wrong)

LoRA.transform has is_expert_linear and is_expert flags, plus normalize_moe_lora (rank
normalization specifically for MoE).
LoRAMerge explicitly has a `# TE Grouped Linear` branch iterating `for i in range(num_gemms)` gemm
by gemm — grouped experts are natively supported.
Even the router can be attached to (LoRATopKRouter).
So "Megatron cannot attach / sglang cannot serve it" — both of those were wrong. Both endpoints
support it out of the box, and each has a measured test against this very model.

Where the real difficulty is (this part still holds)
The hard part is not either endpoint, it is the "in-memory weight conversion + sync contract"
sandwiched in between, inside your own fork — namely
`_convert_qwen3omni_lora_adapter` in qwen3_omni_moe.py and update_lora_from_tensor.py. Right now
those two only understand attention; MLP/MoE raise ValueError across the board.

To extend it, the conversion layer has to emit the layout and naming that sglang's
FusedMoEWithLoRA expects. The concrete pitfalls:

gate/up fusion: linear_fc1's lora_B has to be split/joined in sglang's use_gate_up_lora /
run_gate_up_lora format (not a simple chunk).
Expert structure: sglang uses experts_shared_outer_loras=True (experts share the outer LoRA);
Megatron's TEGroupedLinear likewise shares one adapter across a whole group of gemms (LoRAMerge
uses the same linear_in/out for every weight{i}). The two ends agree conceptually → most likely
one shared adapter per layer rather than 128 independent ones. That actually lowers the
complexity, but it has to be lined up carefully.
Router LoRA (if you attach that too): another matter entirely.

Conclusion (revised)
It is not that "the whole pipeline cannot be pushed through"; it is **an alignment-contract
engineering job**: the capability exists at both ends, and what you fill in is the MoE branch of
the conversion/sync in the middle.
Effort: smaller than I said last time, because you do not have to wrestle with uncontrollable
questions like "can Megatron attach / can sglang serve" — those are already covered by official
tests. What you have to do is write the conversion layer correctly against the MoE LoRA layout
sglang expects (gate_up fusion split + shared-outer expert naming), then validate logprob
alignment following the approach of that test_lora_qwen3_30b_a3b test.
Still-valid advice: first check whether attention-only underfits before deciding to extend; but
"extending to MoE" has none of the hard blockers I described last time.
