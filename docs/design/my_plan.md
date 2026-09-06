# 背景
- slime
    - relax: 支持音频输入
    - miles: 支持lora
- verl
- ms-swift

# 已实现
- Qwen3-Omni thinker + LoRA (qkv, o) 的 speech-to-text offline 翻译（英语音频→中文），reward（sacreBLEU quality）在 100 steps 内随训练上升。

## 参数快照（可回滚基线，改 2a/2b 前固化）
> 回滚点：Relax 分支 `lora-omni-baseline` @ `8bcbb42`（smoke 脚本含续训修复）；
> Modal 入口 `v1 分支 modal_relax_smoke.py::learn_audio`（任务 `s2tt`）。以下为两者叠加后的有效参数。
>
> - 模型/部署：Qwen3-Omni-30B-A3B thinker，bf16；colocate，单机 4×A100-80GB；offload 开（colocate 默认）。
> - 并行：TP=4 / EP=4 / ETP=1 / PP=1 / CP=1；micro-batch=1。
> - LoRA：rank=16，alpha=32，dropout=0.0，name=policy；target=thinker language_model 的 qkv_proj / o_proj。
> - GRPO：advantage=grpo，kl-loss-coef=0，entropy-coef=0，eps-clip=0.2 / eps-clip-high=0.28；max-staleness=0。
> - 优化器：adam，lr=1e-4，lr-decay-style=constant，weight-decay=0，beta=(0.9, 0.95)，optimizer-cpu-offload，precision-aware-optimizer。
> - Rollout：rm-type=bleu，num-rollout=100，rollout-batch=8，n-samples-per-prompt=8，global-batch=64，temperature=1.1，max-response=512，max-prompt=4096，multimodal-keys={"audio":"audios"}。
> - sglang：rollout-num-gpus-per-engine=4，mem-fraction-static=0.7，enable-lora，max-lora-rank=16，max-loras-per-batch=1，lora-target-modules=qkv_proj o_proj，attention-backend=triton，disable-cuda-graph，disable-custom-all-reduce。
> - Megatron misc：attention-backend=flash，no-rope-fusion，dropout=0，grad-allreduce-fp32，attention-softmax-fp32。
> - 存档：save=load=/s2tt/ckpt/s2tt_probe，save-interval=5，max-actor-ckpt-to-keep=1，override-opt_param-scheduler。
> - Modal：4×A100-80GB，retries=10，--detach。

> 每步 raw_reward（BLEU，batch 均值）：
>
> | 步 | BLEU | 步 | BLEU | 步 | BLEU | 步 | BLEU |
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

> | 步 | BLEU | 步 | BLEU | 步 | BLEU |
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

> | 步 | BLEU | 步 | BLEU | 步 | BLEU |
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

- 同传（多轮定长音频块 rollout；整段音频在 env 里按 960ms 切块，每轮喂一块生成增量译文，reward 为整段拼接译文 BLEU）

## 参数快照（相对上面 s2tt 基线的差异；其余同基线）
> Relax 分支 `lora-omni-baseline`：`3a6eb2f`(reward-fix) + `bb7642f`(no-offload)；
> Modal 入口 `v1 分支 modal_relax_smoke.py::learn_simul --tag v2`（save/load=/s2tt/ckpt/s2tt_probe_simul_v2，全新从头训）。
>
> - 同传专属：`--custom-generate-function-path examples.simul_s2tt.rollout.generate`
>   + `--custom-config-path examples/simul_s2tt/config.yaml`（`max_turns=64`, `simul_chunk_ms=960`）。
> - 数据：FLEURS en→zh，卷里 97 条整段音频，平均 9.6s（3.8~23.4s），960ms/块 → 平均 ~10 块/条（4~25）。
> - 部署差异（2a 常驻优化，已并入 simul 脚本）：`--no-offload-train --no-offload-rollout`，
>   `--sglang-mem-fraction-static 0.55`（给常驻 base 腾显存）。base 冻结、两端常驻。
> - 其余（TP4/EP4、LoRA r16/α32 qkv_proj+o_proj、GRPO kl=0、adam lr=1e-4 constant、
>   rm=bleu、rollout-batch=8、n-samples=8、global-batch=64、temp=1.1、max-resp=512、max-prompt=4096）与 s2tt 基线一致。
> - 耗时：~2.8~3.2 min/step（no-offload 后比带 offload 的 ~4.4 min/step 快约 27~35%），无 OOM。

## 两处关键修复
> 1. **reward 去污染**（`3a6eb2f`）：每轮 sglang 返回文本带 `<|im_end|>`，原先直接拼进 `sample.response`
>    喂 BLEU，既多出参考没有的 token、又把跨块 n-gram 全打断。实测（zh tokenizer）BLEU 被压到真实值
>    ~40%（7.2 vs 17.9，≈2.5x）、并抹平组内方差 → advantage≈0、RL 学不动。修法：拼接前正则去 `<|...|>`。
> 2. **no-offload 提速**（`bb7642f`）：2a 的 base 常驻优化此前只在 noffload 脚本，simul 脚本漏了，
>    导致每步照样 offload→wake_up base、耗时无改善。已把两个 flag + mem-fraction 0.55 并入 simul 脚本。

> 每步 raw_reward（batch 均值 BLEU，全新 20 步，reward-fix 后）：
> 注：step K = 第 (K-1) 个 rollout dump（框架 0 索引）。
>
> | 步 | BLEU | 步 | BLEU | 步 | BLEU | 步 | BLEU |
> |----|------|----|------|----|------|----|------|
> | 1 | 0.110 | 6 | 0.211 | 11 | 0.163 | 16 | 0.233 |
> | 2 | 0.064 | 7 | 0.199 | 12 | 0.262 | 17 | 0.264 |
> | 3 | 0.137 | 8 | 0.215 | 13 | 0.205 | 18 | 0.339 |
> | 4 | 0.204 | 9 | 0.220 | 14 | 0.303 | 19 | 0.234 |
> | 5 | 0.158 | 10 | 0.240 | 15 | 0.249 | 20 | 0.231 |
>
> 趋势：前 1/3(step1-7)均值≈0.15 → 后 1/3(step14-20)均值≈0.27，明显上升（min 0.064@step2，max 0.339@step18）。
> 结论：reward-fix 后 RL 确实能学（污染时是 0.06 平躺、advantage≈0）；但这是"整段 BLEU"下的翻译质量提升，
> 非真·同传（缺 read/write 监督、碎块声学孤立编码、reward 无延迟惩罚）——真同传需语义单元切块 / 冷启动 SFT / 延迟奖励。


# todo
- 100 steps ✅
- 权重同步优化（现状：每步都全量推 base + LoRA 每步 unload→reload，两处浪费）：base 冻结，不再 offload/onload，两端常驻 ✅
- 改成同传，即多轮 每轮输入chunk，看 reward 是否上升 ✅
- LoRA 加上 mlp / moe
- 改成 speech-to-speech (训练 thinker)
- 改成异步

# 其他方法
- ms-swift
- verl

（LoRA 加上 mlp / moe）
我上面的判断可能错了。光看文件名就有强烈反证:sglang 里有 lora_moe_runners.py、triton_ops/fused_moe_lora_kernel.py、triton_ops/virtual_experts.py,测试里甚至有 test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py(正是你的模型) 和 test_lora_moe_tp_logprob_diff.py。这直接打脸我"sglang 不支持 MoE LoRA"的说法。让我把关键源码读实。

Megatron 侧的 matcher 默认 target 就含 linear_fc1/linear_fc2,且是按模块名+类型匹配。关键要看它怎么处理 MoE 的 grouped experts。再读 Megatron-Bridge 的 lora.py,以及 sglang 的 lora layers.py 和那个 30B-A3B MoE 测试。

这是关键证据,我之前的判断大部分是错的。看 Megatron-Bridge 的 LoRA:

明确支持 MoE expert:有 is_expert_linear、is_expert 标志、normalize_moe_lora(专门为 MoE 归一 rank),甚至 LoRAMerge.transform 里有 else: # TE Grouped Linear 分支,for i in range(module.to_wrap.num_gemms) 逐 gemm 处理(351-361 行)。
还支持给 router 挂 LoRA(LoRATopKRouter)。
所以"Megatron 能不能挂 grouped expert"根本不是问号——原生支持。让我再确认 sglang 侧对 MoE LoRA 的支持和它期望的命名格式,读那个 30B-A3B 测试和 lora layers。

查清楚了,我得收回之前的判断——那两个"dealbreaker"基本是错的。给你摆事实:

更正:两端其实都原生支持 MoE LoRA
① sglang 侧(我说"几乎肯定不支持"→ 错)

有专门的 FusedMoEWithLoRA 类(layers.py:859)、lora_moe_runners.py、fused_moe_lora_kernel.py。
有一个针对 Qwen3-30B-A3B-Instruct-2507 的 LoRA 回归测试(正是你 thinker 的文本骨干),而且它验证的就是 LoRA logprob 与 trainer 对齐(KL < 5e-3),enable_lora=True + MoE runner 直接跑通。
Engine 参数里有 moe_runner_backend、experts_shared_outer_loras=True 这些 MoE LoRA 专用开关。
② Megatron-Bridge 侧(我说"能不能挂是大问号"→ 错)

LoRA.transform 里有 is_expert_linear、is_expert 标志、normalize_moe_lora(专为 MoE 归一 rank)。
LoRAMerge 明确有 # TE Grouped Linear 分支,for i in range(num_gemms) 逐 gemm 处理——grouped expert 原生支持。
连 router 都能挂(LoRATopKRouter)。
所以 "Megatron 挂不上 / sglang 服务不了" 这两条我说错了,两个端点都现成支持,还各有针对你这个模型的实测。

那真正的难点在哪(这条仍然成立)
难的不是两个端点,而是夹在中间、你自己 fork 里的那套"in-memory 权重转换+同步契约"——也就是 qwen3_omni_moe.py 的 _convert_qwen3omni_lora_adapter + update_lora_from_tensor.py。它俩现在只认 attention,MLP/MoE 一律 raise ValueError。

要扩,你得让转换层产出 sglang FusedMoEWithLoRA 期望的布局和命名,具体几个真实坑:

gate/up 融合:linear_fc1 的 lora_B 要按 sglang 的 use_gate_up_lora / run_gate_up_lora 格式拆/拼(不是简单 chunk)。
expert 结构:sglang 用 experts_shared_outer_loras=True(专家共享 outer LoRA);Megatron 的 TEGroupedLinear 也是一个 adapter 共享给整组 gemm(LoRAMerge 里对每个 weight{i} 用同一个 linear_in/out)。两边理念一致 → 大概率是每层一个共享 adapter,不是 128 个独立的。这反而降低了复杂度,但要对齐清楚。
router LoRA(如果你也挂):又是另一套。
结论(修正版)
不是"整条链路推不动",而是一个**"对齐契约"的工程活**:两端能力都在,你补的是中间转换/同步的 MoE 分支。
工作量:比我上次说的小,因为不用去啃"Megatron 能不能挂 / sglang 能不能服务"这种不可控的东西——它们已被官方测试覆盖。你要做的是照 sglang 期望的 MoE LoRA 布局,把转换层写对(gate_up 融合拆分 + shared-outer expert 命名),然后用那个 test_lora_qwen3_30b_a3b 的思路验证 logprob 对齐。
仍然成立的建议:先验证 attention-only 是否欠拟合再决定要不要扩;但"扩 MoE"本身没有我上次说的那种硬阻塞。