# sglang LoRA × RL(Relax)接入笔记

> 目标:让 **Qwen3-Omni-30B-A3B** 在 RL 训练(Relax / slime / verl)里支持 **runtime LoRA adapter** —— 训练每步把更新后的 LoRA 权重从显存直接同步进 sglang rollout 引擎,不落盘、高频反复。
>
> 本文是两段工作的合并总结:**① 原对话(`cursor_sglang_lora.md`)** 给 Qwen3-Omni 加 LoRA 支持并定位"加载即崩"的 bug;**② 本次** 用 Modal 系统性验证了 sglang 侧全部 LoRA 路径,钉死了 Megatron→sglang 权重契约,并在 Relax 里落地了同步通路。

---

Colocate: 训练和推理共享同一组 GPU
- 推理阶段：SGLang 模型 + KV Cache 在 GPU，Megatron 模型卸载到 CPU
- 训练阶段：Megatron 模型在 GPU，SGLang 模型 + KV Cache 卸载到 CPU
- （如果模型足够小，两份模型都放得下，就不需要卸载）

Async: 训练和推理各自独占不同 GPU
- 不需要卸载
- （双倍 GPU 数量）

base从gpu卸载到cpu再装回gpu，不好

miles没有实现 async/distributed 模式的 LoRA 推送路径（NCCL 广播）
miles直接用 GPU→GPU IPC 把 Megatron 侧的基座权重覆盖进去

---

目的：对 Qwen3-Omni 的 language thinker 做 LoRA RL（GRPO），支持同步（colocate）和异步（async）两种部署模式。

#### ✅ 已完成

| 层次 | 内容 | 状态 |
|------|------|------|
| **训练侧** | Megatron-Bridge LoRA 挂载到 Qwen3-Omni thinker（only LLM，排除多模态塔）；Qwen3-Omni + Megatron LoRA 训练链路能跑；**三层次已全部验证「LoRA 在学」**：机制层（optimizer 真更新 `lora_A/B`、base 冻结）+ 传播层（同步 SGLang 改变生成）+ 效果层（翻译+BLEU，10 步 reward 上升 `[PASS]`），见下文「LoRA 在学：三层次状态」。 | ✅ |
| **命名转换** | Megatron `linear_qkv.adapter` → SGLang `qkv_proj.lora_A/B`，含 GQA reorder | ✅ |
| **LoRA 权重传输** | UpdateLoRAFromTensor 从 Megatron 收集 adapter，并打包成 FlattenedTensorBucket 通过 IPC 发给 SGLang | ✅ |
| **SGLang 热加载** | 在已有/已恢复 base 的前提下，`load_lora_adapter_from_tensors` 只热更新 LoRA adapter；`should_apply_lora` / TP shard / `normalize_qkv_proj` 对 Qwen3-Omni 正确；Qwen3-Omni + SGLang 热加载 LoRA 链路已验证；**修复后生成正常已端到端验证**（坑 23 / `verify_cpu_backup`，2 个 colocate step rollout 全部连贯）。 | ✅ |
| **坑 23 / colocate offload** | `enable_weights_cpu_backup` 缺失导致基座丢失 → 乱码；已修复并端到端验证（offload→resume→热加载后 base 存活、生成连贯） | ✅ |

#### LoRA 在学：三层次状态

| 层次 | 含义 | 状态 |
|------|------|------|
| ① 机制层 | optimizer 真更新 `lora_A/B`，base 冻结不被误动 | ✅ 已验证（`verify_lora_learning.py`，T4：3 步 loss 单调下降，lora_A/lora_B delta 非零、base delta=0、base.grad 全程 None） |
| ② 传播层 | 变化后的 LoRA → 同步 SGLang → 改变生成输出 | ✅ 已验证（「磁盘加载非零 LoRA + 生成」输出与 base 不同） |
| ③ 效果层 | 真实任务 reward 随训练上升 | ✅ 已验证（翻译 zh→en + BLEU，10 步 colocate RL：BLEU 0.46→0.53，`[PASS]`，见下方「效果层验证」） |

> **效果层验证（2026-06-02，翻译 + BLEU，已通过）**：
> - **任务换型**：放弃数学 MCQ（0/1 二值奖励对 30B 太易，要么全对要么全错，零方差），改用 **zh→en 翻译 + 句级 BLEU 连续奖励**。BLEU∈[0,1] 天然有方差，模型几乎拿不到满分 → 有 headroom。
> - **造组内方差的三个杠杆**：① 用**长难句**（从句/口语/多种合理译法），单参考 BLEU 落到 ~0.3-0.6；② **温度 1.3**，让同句多次采样译文分散；③ 数据集扩到 256 降 batch 噪声。
> - **结果（`learn_effect --steps 10 --task translate`）**：基线 step0=0.480；曲线 0.48/0.44/0.47/0.52/0.47/0.55/0.53/0.50/0.50/0.57；**前 3 步均值 0.463 → 后 3 步均值 0.525，Δ=+0.061**；自动判定「有学习信号 + 趋势上升」→ **`[PASS]`**。
> - **代码**：新增 `Relax/relax/engine/rewards/bleu.py`（自包含平滑句级 BLEU，纯标准库，worker 进程可直接用）；`rewards/__init__.py` 注册 `bleu` 到 dispatch + `_SYNC_RM_TYPES`；smoke 脚本 `--rm-type` 改为可配置 `${RM_TYPE}`；`modal_relax_smoke.py` 加 zh→en 翻译数据 + `learn(task="translate")` 默认走翻译+BLEU（温度 1.3、逐样本 `metadata.rm_type=bleu`）。重跑入口：`modal run modal_relax_smoke.py::learn_effect --steps N --task translate`。
>
> <details><summary>历史：数学 MCQ 探查（已被翻译任务取代）</summary>
>
> 修了 `multiple_choice` reward 的真 bug（`extract_answer` 对裸串 label 返回空 → 答对也判 0；已对齐 `openr1mm`：无 `<answer>` 标签则回退裸串 `strip()`）。用 hard 数学 MCQ 跑 colocate RL，确认整条闭环正常、reward 函数对（答对得 1.0）。但 **`raw_reward=1.0`（32/32 全对，零方差）**——Qwen3-Omni-30B thinker + CoT 把 2-3 位乘法做到满分，组内零方差 → advantage=0 → 无梯度。另测「关 thinking / 砍 token」标定：thinking 开=100%、thinking 关或砍到 96 token=0%（伪全错，`<answer>` 被截断），无中间地带 → 砍 `max_response_len` 不可取。最终改用翻译+BLEU 连续奖励解决。
> </details>

> **音频输入验证（2026-06-03，英语语音→中文 S2TT，2 步 smoke 已通过）**：
> - **任务**：移植 slime `examples/my_omni`——英文语音→中文翻译。prompt `<audio>\nPlease translate the English speech into Chinese.`，数据用 **FLEURS**（`google/fleurs`，`en_us` 音频 + `cmn_hans_cn` 文本按 `id` N-way 平行对齐，HF 免费下，`datasets<3`），reward 用 **sacreBLEU(中文 tokenizer)/100**（对齐 my_omni `quality_reward`）。
> - **端到端结果（`learn_effect --steps 2 --task s2tt`，colocate TP=4）**：
>   - prompt 正确渲染 `<|audio_start|><|audio_pad|><|audio_end|>`（用 checkpoint 自带 omni chat_template，非 ChatML 回退）；
>   - 模型从**英语音频**生成连贯中文译文，例：「科学家表示，这次碰撞引发的爆炸非常巨大。」(ref「科学家表示这次碰撞引起的爆炸规模巨大」) → **BLEU 0.324**；「莫尔德医生认为一些病人可能是在医院染上这种病的…」→ **BLEU 0.411**；
>   - 单条 reward 0.23~0.41，与 my_omni ~0.30 baseline 一致、有方差；2 步 `raw_reward 0.230→0.368`，`[PASS]`。
>   - **证明**：音频喂入→audio_tower 编码→thinker 生成、LoRA 热加载、sacreBLEU reward 全链路在 Relax(colocate) 上通了。
> - **修的真 bug（sglang）**：`sglang/srt/models/qwen3_omni_moe.py::get_audio_feature` 把同 batch 里**不同音频**的 `feature/feature_attention_mask` 直接 `torch.cat(dim=0)`，但 RL rollout 下每条样本各自调 processor → mel 帧数不同（936 vs 1056）→ 非拼接维不一致崩。修法：拼接前把 batch 内各 item 右侧 zero-pad 到最大帧数（下游用 `mask.bool()` 抽有效帧，pad 部分 mask=0 不参与，安全）。
> - **代码**：`bleu.py` 扩展支持 dict label `{ground_truth}` + `metadata.tgt_lang` + 装了 sacrebleu 走 zh tokenizer（否则回退字符级）；`modal_relax_smoke.py` 加 `s2tt` 数据卷 + `prep_s2tt`（FLEURS 下载）+ `learn(task="s2tt")`（`MULTIMODAL_KEYS={"audio":"audios"}`、`ROLLOUT_MAX_PROMPT_LEN=4096`）；smoke 脚本接可选 `--rollout-max-prompt-len`。重跑：`modal run modal_relax_smoke.py::prep` 再 `modal run modal_relax_smoke.py::learn_effect --steps N --task s2tt`。

> **同步耗时（colocate，2 步 smoke 实测，含首步 warmup/MoE 内核编译，偏高）**：
> 每步日志 `perf N: {...}`（`train_metric_utils.py:47`）按相打点；`learn()` 现会自动汇总。
>
> | 相 | 含义 | 均值 |
> |----|------|------|
> | `wake_up` | resume：训练模型显存 onload | 8.06s |
> | `sleep` | offload：训练模型显存释放 | 30.58s |
> | `update_weights` | 权重同步（SGLang onload + **LoRA 热推 IPC**） | 14.10s |
> | `train` | 训练 fwd/bwd | 81.07s |
> | `step(总)` | 整步 | 329.17s |
>
> - **同步开销（wake+sleep+update）≈ 52.75s / 329.17s = 16%**；其余 ~195s 是音频 rollout 生成（首步含 MoE 内核编译，偏慢）。
> - **大头是显存 offload/resume（torch_memory_saver，sleep 30s + wake 8s）**，不是 LoRA 热推本身——LoRA adapter 仅 192 个 rank=16 小张量，IPC 推送是亚秒级，`update_weights` 的 14s 主要是 SGLang 权重 onload。结论与预期一致：colocate 同步的代价在「显存搬家」，LoRA 传输几乎免费。稳态（非首步）同步占比会更低。

> **音频输入消融验证（2026-06-04，换配音频对照，已通过）**：
> - **动机**：确认模型「真在听音频」，而非靠语言先验/数据泄漏瞎猜。
> - **方法**（`verify_audio_ablation`，2×A100-80GB TP=2 推理、不训练，约 15 分钟）：取 2 对 S2TT 样本，prompt 文字完全相同（仅「把英语语音翻成中文」），交叉喂音频：correct=原配音频，swapped=对方音频，看输出是否跟着音频内容变。
> - **结果（决定性 `[PASS]`）**：输出**严格跟随音频内容**——
>   - pair0：A 音频(卫星通话)→「…你正在使用卫星」BLEU 0.435；把 A 的 prompt 配上 **B 的音频**(爪哇菜)→ 输出立刻变「爪哇菜在群岛各地并不普遍…」，对 A 参考 BLEU 崩到 **0.004**。
>   - 2 对、4 个交叉组全部「换音频→输出跟着换」。
> - **结论**：音频确实被编码并驱动生成（不是语言先验）。**音频输入链路确认无误**。
> - **踩坑提醒**：standalone SGLang 启动器必须复用训练路径已填平的坑——`attention_backend="triton"`（绕 flashinfer 版本检查）、`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（绕 sglang-kernel 检查）、prompt 手拼 ChatML（根 tokenizer 无 chat_template，别用 `apply_chat_template`）。

> **正式音频 RL 实验（2026-06-04，en 语音→zh 文本 + BLEU，40 步，已通过）**：
> - **配置**：`learn(task="s2tt")`，4×A100-80GB colocate TP=4；FLEURS en→zh；`n_samples_per_prompt=8`、`global_batch=64`、`temperature=1.1`、`lr=1e-4`；BLEU(sacreBLEU zh) 连续奖励。
> - **结果（`[PASS]` 效果层）**：BLEU 随训练上升——
>   - 首 1/3 均值 ≈ **0.295**（baseline）→ 末 1/3 均值 ≈ **0.370**，**Δ ≈ +0.075 BLEU（相对 +25%）**；区间 0.234~0.431，末段 step 37/38/39 = 0.431/0.414/0.418 稳在高位。
>   - 旁证：response 长度变短（28→20 token）、log_probs 上升（生成更自信）；抽样 `东非岛屿位于非洲东海岸外的印度洋上`（BLEU 0.659），译文连贯且正确。
>   - 全程 40 步零报错。
> - **每步 raw_reward（BLEU，batch 均值）**：
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
>
> （步 1–40 对应日志 `rollout_id` 0–39，指标为每步 `rollout/raw_reward` batch 均值。）
> - **同步耗时（稳态，每步 ~155s，比 2 步 smoke 的首步 warmup 数据更可信）**：
>
> | 相 | 含义 | 均值 |
> |----|------|------|
> | `train` | 训练 fwd/bwd | 92.2s |
> | `sleep` | offload：训练模型显存释放 | 25.7s |
> | `update_weights` | 权重同步（SGLang onload + LoRA 热推 IPC） | 12.1s |
> | `wake_up` | resume：训练模型显存 onload | 8.0s |
> | **同步开销合计** | wake+sleep+update | **≈45.7s ≈ 整步 29%** |
>
> - 大头仍是显存 offload/resume（sleep 25.7s + wake 8.0s），LoRA 热推本身亚秒级；与 2 步 smoke 结论一致：colocate 同步代价在「显存搬家」，LoRA 传输几乎免费。

Qwen3-Omni + SGLang + TP: 热加载 LoRA + 生成正常 ✅
Qwen3-Omni + SGLang + TP: 加载 merge（磁盘加载 PEFT adapter）+ 生成正常 ✅
Qwen3-Omni + colocate RL（train→offload→resume→热加载→推理）: 2 step rollout 全部连贯，base 在 offload/resume 后存活 ✅

验证结论（Qwen3-Omni + SGLang TP=4）：

| 测试项 | 结果 |
|--------|------|
| Base 模型生成 | ✅ 正常输出 `"2 + 3 = 5."` |
| 热加载 LoRA（从 tensor，零初始化 B）+ 生成 | ✅ 正常，输出与 base 一致（符合预期：零 B = 无影响） |
| 磁盘加载 PEFT adapter（零初始化 B）+ 生成 | ✅ 正常，输出与 base 一致 |
| 磁盘加载非零 LoRA + 生成 | ✅ 输出与 base 不同，证明 LoRA 权重确实被应用 |

relax的omni + miles的lora
qwen3 omni thinker(语言模型，非多模态塔) 加lora


修正后的完整对比：

| 维度 | Miles LoRA colocate | Relax LoRA colocate |
|------|---------------------|---------------------|
| 卸载什么 | 只卸载 KV cache + CUDA graphs | 全部卸载（weights + KV + 一切） |
| Base weights | 始终留在 GPU，永不卸载 | 卸载到 CPU，恢复时 CPU→GPU 拷回 |
| 第一次同步 | GPU→GPU 拷 base 到 SGLang buffer（一次性） | 不需要（CPU backup 会恢复） |
| 后续每步 | 只推 LoRA（~50MB），base 从不动 | CPU→GPU 恢复 base（~60GB）+ 推 LoRA |
| 需要 CPU backup？ | ❌ 不需要（weights 没被卸载） | ✅ 必须（weights 被卸载了） |
| patch_param_grad_buffer | 确保训练侧 LoRA 参数也不被误卸载 | 没有 |

所以 Miles 的真正做法是：LoRA 模式下 base weights 根本不卸载，一直在 GPU 上。

base怎么处理？
IPC?
GPU↔GPU: NVLink
CPU↔GPU: PCIe


## Miles LoRA（colocate 同步机制，代码核实）

> 交接提示：本节只讲 Miles。结论一句话——**Miles LoRA 仅支持 colocate；colocate 下每步「全量 base 走 CUDA IPC 重推 + LoRA 走热加载」，因此不依赖 cpu_backup，并通过训练期卸载 sglang base 省显存。**

### 支持矩阵

| 模式 | 支持 | 依据 |
|------|------|------|
| colocate | ✅ | `actor.py:161` `if colocate: UpdateWeightFromTensor`，`is_lora=True` 全程生效 |
| async / disaggregated | ❌ | 见下「async 不支持的两条铁证」 |

### 同步方式（base 和 lora，colocate 下）

   

核心代码（`update_weight_from_tensor.py:192-221`）：

```python
# For LoRA+distributed: base weights are frozen, skip after first round.
if not (self.is_lora and self.use_distribute and self._lora_base_synced):
    for hf_named_tensors in ...weight_type="base":
        refs, long_lived_tensors = self._send_base_params(hf_named_tensors)
if self.is_lora:
    for hf_named_tensors in ...weight_type="lora":
        refs, long_lived_tensors = self._send_lora_params(hf_named_tensors)
    if self.use_distribute and not self._lora_base_synced:
        self._lora_base_synced = True
```

- LoRA 只能在 `use_distribute=False`（纯 colocate）下跑（否则 `_send_lora_params` raise）。
- 既然 `use_distribute=False`，base 的跳过条件 `is_lora and use_distribute and _lora_base_synced` 恒为 False → **base 每步都推**；且 `_lora_base_synced` 只在 `use_distribute` 时才置 True → 纯 colocate 下永远 False。所以那段「skip after first round」对 LoRA 是**死代码**。
- base：每步 `_send_base_params` → `_send_to_colocated_engine`（else 分支）→ `update_weights_from_tensor`（CUDA IPC，全量基座）。
- lora：每步 `_send_lora_params` → `_send_to_colocated_engine`（is_lora 分支）→ 先 `unload_lora_adapter` 再 `load_lora_adapter_from_tensors`（热加载）。

### async 不支持的两条铁证

1. `update_weight_from_tensor.py:263-264`：混合（含分布式引擎）时 LoRA 分支直接 `raise NotImplementedError("LoRA weight sync is not yet supported for distributed (non-colocated) engines")`。
2. 纯分布式类 `UpdateWeightFromDistributed`（`broadcast.py`）和 `UpdateWeightP2P`（`p2p.py`）虽然接收 `is_lora` 参数但**根本不存、不分支** —— `__init__` 里没有 `self.is_lora`，`update_weights` 只做全参 gather+broadcast/P2P。即非 colocate 跑 LoRA 不会报错但行为是错的（把 lora_A/B 当普通权重传）。

### 关键文件 / 类 / 函数

| 作用 | 位置 |
|------|------|
| 选择 updater | `miles/backends/megatron_utils/actor.py:161-174` |
| 主体 | `update_weight/update_weight_from_tensor.py` → `UpdateWeightFromTensor.update_weights` / `_send_base_params` / `_send_lora_params` / `_send_to_colocated_engine` |
| async 拦截 | 同文件 `:264` `NotImplementedError` |
| 分布式（无 LoRA） | `update_weight_from_distributed/broadcast.py`（`UpdateWeightFromDistributed`）、`p2p.py`（`UpdateWeightP2P`） |

### 省显存？

关键默认值（`miles/utils/arguments.py`）：

```python
parser.add_argument(
    "--offload-rollout-level",
    type=str,
    nargs="+",
    default=["kv_cache", "weight"],
    ...
)
```

默认 `["kv_cache", "weight"]` —— 所以 Miles colocate 默认连 sglang 的 base weight 一起卸载。（LoRA 示例把这行注释掉了，注释掉 = 用默认值 = 仍然卸 weight。）

**答案：能省显存，但要分清「省」来自哪。**

- 省显存来自「训练时卸载 sglang base」，**不是**来自「IPC 重推」本身。
- `train.py` 里 `if "weight" in offload_rollout_level: offload_tags.append(WEIGHTS)` → 训练阶段 sglang 的 base（~15GB/卡）被 release 掉，GPU 只剩 Megatron（frozen base + LoRA optim + activations）。
- IPC 重推的真正作用是「让卸载变安全」：因为每步都能从训练侧 GPU 上那份 base 重新 IPC 灌回 sglang，所以 sglang base 可以放心丢、不需要任何备份。重推不是为了省显存，而是为了在「丢了 base」之后还能正确恢复。

| 维度 | Miles（卸 + IPC 重推） |
|------|------------------------|
| 训练时 sglang base | 释放，省 ~15GB/卡 |
| 额外 CPU RAM | 0 |
| base 恢复 | 训练侧 GPU → 同卡 IPC 灌回 |

---

miles的做法：megatron bridge不支持qwen3 omni

### Miles 式 LoRA 需要的组件 × Qwen3-Omni 支持状态

> Miles 式 = colocate，每步「base 全量 CUDA IPC 重推 + LoRA 热加载」，不依赖 cpu_backup。

#### A. Megatron 训练侧

| # | 组件 | Qwen3-Omni 状态 | 在哪 |
|---|------|----------------|------|
| 1 | LoRA 挂到 thinker（排除 audio/visual/talker） | ✅ | Relax `model_provider.py` scoped target，真 30B 验证命中 |
| 2 | Qwen3-Omni thinker（MoE）模型定义 | ✅ Relax / ❌ bridge | Relax 自建 provider；bridge 只有 Qwen2.5-Omni |
| 3 | LoRA 优化器更新 lora_A/B | ✅ 三层全验证：机制层（`verify_lora_learning.py`：lora_A/B delta 非零、base 冻结）+ 传播层 + 效果层（翻译+BLEU 10 步 reward 上升 `[PASS]`） | `verify_lora_learning.py` / `learn_effect` |

#### B. 权重转换 Megatron→HF

| # | 组件 | Qwen3-Omni 状态 | 在哪 |
|---|------|----------------|------|
| 4 | LoRA adapter 命名转换（qkv/o_proj.lora_A/B + GQA reorder） | ✅ | `qwen3_omni_moe.py` LoRA 段 + `_reorder_qkv_lora_b` |
| 5 | base 全量转换（含 MoE expert/shared/router/attn/norm） | ✅ 已覆盖 | `convert_qwen3omni_to_hf` |
| 6 | bridge `export_adapter_weights` 路径 | ❌ | 老 fork 无此 API + bridge 无 Qwen3-Omni 映射 |

#### C. 传输（Megatron→sglang）

| # | 组件 | Qwen3-Omni 状态 | 在哪 |
|---|------|----------------|------|
| 7 | LoRA 热加载 IPC（`load_lora_adapter_from_tensors`） | ✅ 已验证 TP=4 | `UpdateLoRAFromTensor` |
| 8 | base 全量 CUDA IPC 重推（`update_weights_from_tensor`） | ⚠️ 传输层在，Omni base 未验证 | `UpdateWeightFromTensor._send_to_colocated_engine` |
| 9 | MoE expert 的 TP/EP gather（base 重推时） | ⚠️ 未验证，最大风险 | — |

#### D. sglang 侧

| # | 组件 | Qwen3-Omni 状态 | 在哪 |
|---|------|----------------|------|
| 10 | Qwen3-Omni 模型 + LoRA 闸门（`_lora_pattern` / `should_apply_lora` / `normalize_qkv_proj`） | ✅ 已验证 | 你的 sglang fork |
| 11 | `load_lora_adapter_from_tensors` 接收 | ✅ 已验证 | sglang |
| 12 | `update_weights_from_tensor` 接收 base | ✅（model-agnostic） | sglang |

#### E. colocate 内存编排

| # | 组件 | Qwen3-Omni 状态 | 在哪 |
|---|------|----------------|------|
| 13 | offload/resume + memory_saver | ✅ | Relax rollout manager |
| 14 | 「base 靠每步 IPC 重推、不用 cpu_backup」 | ⚠️ 逻辑可行但未接线 | Relax 当前用 cpu_backup 变体，需小改 |

**汇总**：已支持 Omni（Relax direct）→ 1, 2, 3（机制层+效果层，翻译+BLEU 已验证 reward 上升）, 4, 5, 7, 10, 11, 12, 13。不支持/未验证 → ❌ 整条 bridge 路径（6）；⚠️ base 全量重推 + MoE gather（8, 9）；⚠️ Miles 式接线（14）。硬骨头：**#9 MoE expert 的 TP/EP gather**。

---

list:

- 命名转换
- 同步方式
- sglang (lora: miles, verl | omni: relax)
- megatron (lora: miles, verl | omni: relax)


---

① Megatron 训练 LoRA（优化器更新 lora_A/B）
       │   ← slime 发现 MoE+PEFT 下 lora_B 可能不被更新（深坑，训练侧，另说）
       ▼
② 跨 4 卡 TP 收集 adapter → gather 成整张
       │   ← ✅ gather 逻辑已在 T4 验证正确（Relax/Slime 两种方法数值一致）
       │   ← ✅ .contiguous() 修复对非连续 buffer 有效
       │   ← ✅ torch_memory_saver 场景已验证: CPU备份→GPU→gather 正确; translate_gpu_to_cpu=True 是正确修复
       ▼
③ Megatron 命名 → HF/PEFT 命名（qkv/o_proj.lora_A/B）  ← ✅ 已在 T4 验证正确
       ▼
④ 序列化 + IPC 推给 sglang                            ← ✅ 4xA100 smoke 已通（pickle.dumps + base64）
       ▼
⑤ sglang load_lora_adapter_from_tensors 热加载         ← ✅ 全 4 TP "completes"
       ▼
⑥ sglang forward 真正用上 LoRA（rollout 推理）         ← ✅ 32 samples, 308 tok/s
       ▼
⑦ 训练 forward（PPO loss 计算）                        ← ✅ 坑 21 已修复，2 step GRPO 训练完成（loss=0 符合预期）

---

## TL;DR(简短版)

| 阶段 | 做了什么 | 结论 |
|---|---|---|
| 原对话 | 给 Qwen3-Omni 加 `_lora_pattern` + `should_apply_lora` 闸门(只挂 thinker,挡 audio/visual/talker);在 v0.5.9 容器里复现"加载 LoRA 后 base 输出立刻崩成 stop token" | 怀疑动态加载路径有问题,但因 **GPU 占满 + 本地代码是 main 比容器新** 卡住 |
| 本次-环境 | 改用 **Modal** 按需起 A100-80GB;关键修复:**本地 sglang 是 `main`(非 v0.5.9),用 `lmsysorg/sglang:latest` + 整份本地源码 PYTHONPATH 覆盖**,彻底消除版本错配 | 环境干净可复现 |
| 本次-实验 E | 启动时静态 `--lora-paths` 注入 | ✅ base==lora,路径干净 |
| 本次-实验 F | 动态 `/load_lora_adapter` | ✅ **main 已修旧 bug**(旧容器特有) |
| 本次-实验 G | in-process Engine 模拟 RL 热更新循环 | ✅ 生效/可逆/连续 N 步稳定 |
| 本次-实验 H | `flattened_bucket` 快速同步(in-process) | ✅ 与普通路径逐字符一致 |
| 本次-实验 I | Megatron fused-qkv vs PEFT split-qkv 契约 | ✅ **logprob 逐位差 0**,fused 可直接用 |
| 本次-实验 J | **Relax 真实链路**:HTTP server + CUDA IPC 序列化 POST `/load_lora_adapter_from_tensors` | ✅ 端到端通,未踩 verl #4065 序列化 bug |
| 本次-落地 | Relax `sglang_engine.py` 加 LoRA 同步方法(Block 1);启动参数走 `--sglang-*` 透传(Block 2,零代码) | ✅ 传输层已验证 |
| 本次-验证 K | **Megatron-Bridge PEFT 实测**(`verify_lora_attach.py`,Modal + slime 镜像 + T4) | ✅ 命名/冻结/scoping 全确认 |
| 本次-Block 4 | Relax 训练侧挂 LoRA(`model_provider.py` + `arguments.py` + `model.py`) | ✅ 已实现 |
| 本次-Block 3 | Megatron adapter 权重→sglang 改名打包热推(`qwen3_omni_moe.py` + `update_lora_from_tensor.py` + `hf_weight_iterator_direct.py` + `actor.py`) | ✅ 已实现 |
| 本次-验证 L | **Block 3 转换器纯函数实测**(`verify_lora_convert.py`,本地 CPU torch) | ✅ 命名/形状/qkv 重排逐位对(以 base 拆分为 ground truth) |
| 本次-验证 M | **TP=2 下 adapter 切分复原实测**(`verify_lora_tp.py`,Modal 2×T4) | ✅ 4 个 adapter 全被 TP 切分,gather 按全局行号逐位拼对,TP 风险清零 |
| 本次-Block 5 | rollout 请求带 `lora_path` 引用热推的 adapter(`sglang_rollout.py` + `arguments.py`) | ✅ 已实现,链路闭合 |

**一句话:5 个 Block 全部落地(sglang 侧 + 传输层 + 训练侧挂 LoRA + adapter 同步 + rollout 引用),整条 RL runtime LoRA 链路已打通。剩下只是真集群上的端到端联调。**

---

## 详细版

### 1. 背景与架构

RL rollout 循环:
```
训练一步 → 新 LoRA 权重(GPU tensor) → 同步进 sglang → 用新权重 rollout 采样 → 继续训练 → 循环
```
与"磁盘加载一次"不同:**高频、内存直传、反复覆盖同一 adapter**。

Qwen3-Omni 是多模态模型(thinker + audio_tower + visual + talker),**LoRA 只挂在 thinker 这个文本 LLM 上**,其余模块必须排除。

#### 1.1 训练/部署拓扑(rollout 引擎和训练放在哪)

训练方式要拆成**两个正交维度**看,别和"权重怎么搬"混为一谈。

**维度 A:部署拓扑**

| 拓扑 | 含义 | 代价 |
|---|---|---|
| **Colocate(同卡/混部)** | 训练(Megatron)和 rollout(sglang)**共享同一批 GPU**,分时复用:训练时把推理 offload/sleep,rollout 时反过来 | 省卡,有切换开销 |
| **Disaggregated(分离部署)** | 训练一批卡常驻,rollout 引擎另一批卡常驻 | 吞吐高、流水好,但要更多卡 |

**维度 B:同步性**

| | 含义 |
|---|---|
| **同步(on-policy)** | 训练一步 → 推权重 → rollout 采样 → 再训练,串行 |
| **异步(`fully_async`/off-policy)** | rollout 和训练并行跑,权重周期性推送(Relax 有 `fully_async` 开关) |

三框架都支持 colocate + disaggregated;区别在成熟度:slime 以 colocate 起家,verl 两种都强,Relax 两种都有 + `fully_async`。

#### 1.2 权重同步的传输方式(含 HTTP 澄清)

**一句话:控制面(发指令/元数据)和数据面(搬几十 GB 权重)是分开的;HTTP 只在控制面,权重 bytes 从不过 HTTP。**

sglang rollout 引擎是个**独立进程**,两种接入形态:
- **HTTP server 模式**:暴露 REST API(`/update_weights_from_tensor`、`/load_lora_adapter_from_tensors`、`/generate`…)。Relax `sglang_engine.py` 即此,`_make_request` 就是 HTTP POST。
- **in-process Engine 模式**:训练进程里直接持有 sglang `Engine` 对象,函数调用,连 HTTP 都省(实验 G/H 用的就是这个)。

数据面按"权重 bytes 实际怎么传"分三种:

| 方式 | 适用 | 数据面 | 控制面 | Relax 实现 |
|---|---|---|---|---|
| **① CUDA IPC** | 同机 colocate | 训练侧把 GPU tensor 序列化成 **CUDA IPC handle**(指向显存的句柄,非数据本身),rollout 进程拿 handle **直接 map 同一块显存**读 | 几 KB 的 handle + 元数据走 HTTP/Ray | `UpdateWeightFromTensor` / **LoRA 的 `UpdateLoRAFromTensor`** |
| **② NCCL broadcast** | 跨机 disaggregated | 建 NCCL 组(训练 rank0 + 所有 engine GPU),权重 **GPU→GPU 直传**(NVLink/IB),不过 CPU/HTTP | HTTP/Ray 只做握手 + 发"广播哪些权重"的指令 | `UpdateWeightFromDistributed` |
| **③ 落盘 / safetensors** | 启动时静态加载 | 训练存 ckpt,rollout 从**磁盘 load**,round-trip 慢 | — | LoRA 的 `load_lora_adapter`(from path) |

> **为什么 LoRA 的 `load_lora_adapter_from_tensors` 走 `_make_request`(HTTP)却不慢**:HTTP 请求体里装的是 **CUDA IPC handle + PEFT config**(几 KB),adapter 权重本体经同机共享显存(IPC)给到 sglang。所谓"HTTP 同步"准确说是"**HTTP 信令 + CUDA IPC 数据**"。

#### 1.3 三框架横向对比

| 框架 | 训练后端 | 主拓扑 | 同步传输 |
|---|---|---|---|
| **slime** | Megatron | colocate 为主 | IPC(`flattened_bucket`)、HTTP server 模式;本项目 LoRA 思路借鉴 slime-rl |
| **verl** | FSDP / Megatron | colocate + disaggregated 都强 | NCCL / IPC 都有(注意 verl #4065 的 IPC 序列化 bug,实验 J 验证未踩到) |
| **Relax**(本项目) | Megatron | colocate + disaggregated + `fully_async` | `UpdateWeightFromTensor`(IPC)/ `UpdateWeightFromDistributed`(NCCL) |

> **本项目 LoRA 走的是 ① colocate + CUDA IPC + HTTP 信令**(`UpdateLoRAFromTensor`)。这也是为什么目前只支持 colocate;NCCL(②)那条 LoRA 路径尚未接(留作扩展)。

### 2. 原对话的改进(`cursor_sglang_lora.md`)

#### 2.1 两个 patch(本地 sglang 源码)

- **`python/sglang/srt/models/qwen3_omni_moe.py`**:`_lora_pattern` 正则,只匹配 `thinker.model.layers.*.self_attn.{qkv_proj,o_proj}` / `mlp.experts` / `lm_head` / `embed_tokens`,排除 `audio_tower` / `visual` / `talker`。
- **`python/sglang/srt/lora/lora_manager.py`**:`should_apply_lora` opt-in 闸门 —— 模型可声明哪些 module 参与 LoRA,`lora_manager` 在初始化时按此过滤。

> 离线校验脚本:`sglang/test/srt/lora/_smoke_offline.py`(不 import sglang,纯 AST + 正则验证 patch 在位,Windows 也能跑)。

#### 2.2 定位的 bug

在偏旧的 sglang 容器(v0.5.x/0.5.9)里,**动态 `/load_lora_adapter` 加载 adapter 后,即使 LoRA 权重 B=0(数学上 no-op),base 输出也立刻崩成 `<|im_end|>` stop**。当时怀疑是动态加载后 batch_info / permutation 状态没刷新。

#### 2.3 当时的拦路坑

- 本地代码来自 **GitHub main**,比容器里的 sglang 新 → 只 cp 两个文件会撞接口(`FusedMoEWithLoRA` 缺失、`virtual_experts.py` 依赖 `jit_kernel.moe_align` 缺失,需 stub)。
- 服务器 8 张 GPU 全占满,起不了 server。

### 3. 本次改进

#### 3.1 环境:Modal + 正确的版本对齐(关键)

**根因诊断**:本地 `sglang/` 是 `main` 分支(2026-05 提交,需 CUDA 13 / sgl-kernel 0.4.x / flashinfer 0.6.11),**不是 v0.5.9**。之前所有"新代码 + 旧容器"的坑都源于此。

**解法**(见 `modal_run.py`):
- 基础镜像用 `lmsysorg/sglang:latest`(从 main 构建,二进制依赖匹配)。
- **不只 cp 两个文件**,而是把整份本地 `sglang/python` 通过 `add_local_dir` 进镜像 + `PYTHONPATH` 覆盖 → `import sglang` 整体指向本地源码,所有文件版本一致,不再 stub。
- 模型权重下到 **Modal Volume** 缓存,二次运行秒级命中。
- 容器内 `_verify_local_source()` 断言 `import sglang` 来自覆盖路径 + 校验闸门 + `FusedMoEWithLoRA` 可导入。

#### 3.2 实验 E–J(全部在 `modal_run.py`,单卡 A100-80GB)

| 实验 | 入口 | 验证内容 | 方法要点 | 结论 |
|---|---|---|---|---|
| E | `modal run modal_run.py` | 静态 `--lora-paths` 注入 B=0 adapter | server 启动即带 adapter,对比 base vs lora | base==lora,32 token 正常 |
| F | `modal run modal_run.py --dynamic 1` | 动态 `/load_lora_adapter` | server 起来后 API 加载再测 | base==lora,**旧 bug 在 main 已消失** |
| G | `modal run modal_run.py::rl` | RL 热更新循环 | in-process `sgl.Engine`,反复 `load_lora_adapter_from_tensors` 推新版本,LRU 淘汰 | 生效(非零≠base)+ 可逆(B=0==base)+ 8 步稳定不泄漏 |
| H | `modal run modal_run.py::flat` | `flattened_bucket` 快速路径 | 同一份权重分别走逐 tensor / FlattenedTensorBucket | 输出逐字符一致,无偏差 |
| I | `modal run modal_run.py::mega` | Megatron fused-qkv == PEFT split-qkv | 同一底权重造 fused 和 split 两版,比 **logprob** | token id 全同,logprob 差 `0.000e+00` |
| J | `modal run modal_run.py::relax` | **Relax 真实传输链** | HTTP server + GPU tensor → FlattenedTensorBucket → MultiprocessingSerializer(CUDA IPC)→ POST `/load_lora_adapter_from_tensors` | 端到端成功,未踩 CPU-tensor 序列化 bug(verl #4065) |

**实验设计教训**:
- 用原始 `text` 补全测 instruct 模型会立刻吐 `<|im_end|>`(空输出)→ 必须套 Qwen chat 模板(`_chat_prompt`)。
- 比"文本是否相同"会被"都退化成空"骗到 → **比 token id + logprob** 才严谨(实验 I 的做法)。
- IPC 序列化要求 tensor **在 GPU 上**(RL 实际如此),CPU tensor 会触发 sglang `patch_torch.py` 的 bug。

#### 3.3 Megatron → sglang 权重契约(实验 I 钉死)

sglang 的 `normalize_qkv_proj`(`lora.py:213`)接受两种 qkv 格式,**fused 分支正好对上 Megatron**:

| Megatron-Bridge 名 | sglang 期望名 | 形状 |
|---|---|---|
| `decoder.layers.N.self_attention.linear_qkv.adapter.linear_in` | `…thinker.model.layers.N.self_attn.qkv_proj.lora_A.weight` | `[r, hidden]`(sglang 自动 `repeat(3,1)`) |
| `…linear_qkv.adapter.linear_out` | `…qkv_proj.lora_B.weight` | `[q+2kv, r]`=`[5120, r]`(已完整,直接用) |
| `…linear_proj.adapter.linear_in` | `…o_proj.lora_A.weight` | `[r, 4096]` |
| `…linear_proj.adapter.linear_out` | `…o_proj.lora_B.weight` | `[2048, r]` |

要点:`adapter.linear_in`=lora_A,`adapter.linear_out`=lora_B;`decoder.layers`→`thinker.model.layers`;**不需要拆 q/k/v**;tp_size>1 时 lora_B 输出维要按 TP rank 拼回 `[5120, r]`。
adapter_config:`target_modules=["qkv_proj","o_proj"]`,`r=32`,`lora_alpha=…`。

#### 3.4 Relax 落地

- **Block 1(已做)** `relax/backends/sglang/sglang_engine.py` 新增:
  - `load_lora_adapter_from_tensors(lora_name, serialized_tensors, config_dict, load_format, pinned)` —— POST `/load_lora_adapter_from_tensors`,payload 对齐 sglang `LoadLoRAAdapterFromTensorsReqInput`
  - `load_lora_adapter(name, path)` / `unload_lora_adapter(name)`
  - 全部复用现有 `_make_request` HTTP 模式;传输链由实验 J 实测验证。
- **Block 2(零代码)** Relax 用 wrapper 把 `ServerArgs.add_cli_args` 全字段加 `--sglang-` 前缀(`arguments.py`),`_compute_server_args` 透传。开 LoRA 只需启动脚本加:
  ```
  --sglang-enable-lora --sglang-max-lora-rank 32 \
  --sglang-lora-target-modules qkv_proj o_proj \
  --sglang-max-loras-per-batch 2 --sglang-max-loaded-loras 2
  ```

#### 3.5 验证 K:Megatron-Bridge PEFT 实测(`verify_lora_attach.py`)

在接 Block 4 之前,用真实 Megatron + Megatron-Bridge + TE 环境(借 slime 预构建镜像 `slimerl/slime:latest`,Modal + **T4**,成本几分钱)实测了 4 件事,**避免写代码后才发现返工**:

| 检查 | 结果 | 结论 |
|---|---|---|
| 1 adapter 命名 | ✅ | 实测打印 `...self_attention.linear_qkv.adapter.linear_in/out.weight`,与 §3.3 契约一致 |
| 2 冻结 | ✅ | 普通 `LoRA` 已冻结全部基座、仅 adapter 可训 → **不用 `VLMLoRA`** |
| 3 过匹配(暴露风险) | ✅ | 裸 `target=["linear_qkv"]` 会把 LoRA **误挂到音频塔** |
| 4 限定 | ✅ | 改 `["*language_model*linear_qkv", ...]` 后只挂 language_model,音频塔 0 个 |

**两个会导致返工的纠正**:
- ❌ 不用 `VLMLoRA`(其 `freeze_model` 按 LLaVA 属性 `model.vision_model/language_model` 写死,Qwen3-Omni 对不上)→ ✅ 用普通 `LoRA`(冻全部 + adapter 可训)。
- ❌ 裸 target 会误挂多模态塔 → ✅ 必须用通配符限定到 `language_model`。

> 注:Megatron-Bridge 自带的是 Qwen2.5-Omni(`models/qwen_omni/qwen25_*`),无 Qwen3-Omni 原生 bridge —— 但 PEFT 模型无关,Relax 自己能建模型,故不影响。
> 运行:`modal run modal_verify_lora.py`(脚本 `verify_lora_attach.py`)。

#### 3.6 Block 4 落地(已实现)

thinker 上挂 LoRA,改了 3 个文件(均在 `Relax/relax/`):

- **`utils/arguments.py`**(`add_train_arguments`):新增 CLI
  - `--lora-enable`、`--lora-rank`(默认 32)、`--lora-alpha`(默认 32)、`--lora-dropout`(默认 0)
  - `--lora-target-modules`,**默认 `["*language_model*linear_qkv", "*language_model*linear_proj"]`**(已按验证 K 限定 thinker)
- **`backends/megatron/model_provider.py`**:新增 `apply_lora_to_model()` 和 `wrap_model_provider_with_lora()`。在 provider 建出 GPTModel 后、DDP 包裹前调 `LoRA(...)(model, training=True)`。
- **`backends/megatron/model.py`**(`setup_model_and_optimizer`):`args.lora_enable` 为真时用 LoRA wrapper 替代通用 freeze wrapper(PEFT 自带冻结)。

启动脚本开 LoRA(训练侧):
```
--lora-enable --lora-rank 32 --lora-alpha 32
# --lora-target-modules 一般用默认即可
```

#### 3.7 Block 3 落地(已实现)

把训练侧的 adapter 权重**热推**到 colocate 的 sglang rollout 引擎。**核心思路:复用全量同步已有的 PP/EP broadcast + TP all-gather,只是把范围缩到 adapter、改名、改发送 API。**

- **`weight_conversion/qwen3_omni_moe.py`**:在 decoder 层分支里新增 `.adapter.` 处理,`_convert_qwen3omni_lora_adapter()` 做命名转换:
  - `linear_qkv.adapter.linear_in` → `base_model.model.thinker.model.layers.N.self_attn.qkv_proj.lora_A.weight`(形状 `[r, hidden]`,**sglang 自动 repeat 3 份**)
  - `linear_qkv.adapter.linear_out` → `...qkv_proj.lora_B.weight`(形状 `[qkv_out, r]`,**必须用与 base 权重相同的 view/split/reshape 把分组交错排布重排成 `[q;k;v]`** —— 见 `_reorder_qkv_lora_b()`)
  - `linear_proj.adapter.linear_in/out` → `...o_proj.lora_A/B.weight`(直接改名,无需重排)
- **`weight_update/hf_weight_iterator_direct.py`**:`HfWeightIteratorDirect` 加可选 `name_filter`,只保留 `.adapter.` 的全局参数名 → 不会 gather/转换整座 30B 基座。
- **`weight_update/update_lora_from_tensor.py`(新)**:`UpdateLoRAFromTensor`
  - 复用 direct 迭代器(filter=adapter)做 TP/PP 复原 → 每个 rank 都拿到**完整** adapter 张量;
  - 各 colocate 引擎的 gather 源 rank 把全部 adapter 打成**一个** `FlattenedTensorBucket`,经 IPC 调 `load_lora_adapter_from_tensors`(而非 `update_weights_from_tensor`);
  - RL 循环里同名 adapter 不能重复加载 → **先 `unload` 旧的再 `load` 新的**,固定 `lora_name`(默认 `"policy"`)供 Block 5 引用;
  - `config_dict` 用标准 PEFT 字段(`r`/`lora_alpha`/`target_modules=[q/k/v/o_proj]`),sglang 内部把 q/k/v 归一到 qkv_proj。
- **`backends/megatron/actor.py`**:`args.lora_enable` 为真时用 `UpdateLoRAFromTensor`(仅 colocate);并跳过 `ci_test` 下的 base `weight_version` 一致性校验(LoRA 不改基座版本)。

**当前边界**:仅支持 **colocate(IPC)**;分布式(NCCL)LoRA 推送、PP>1 已含 broadcast 但未在多卡实跑验证。**TP>1 已由验证 M 实测清零** —— 实测 Bridge 把 4 个 adapter 全做了 TP 切分(连 lora_A 的 rank 维都切),都带正确的 `tensor_model_parallel`/`partition_dim`,复用 `all_gather_params_async` 按全局行号逐位拼对。即「gather 复原(验证 M)→ Block 3 qkv 重排(验证 L)」两段串联端到端正确。

#### 3.8 Block 5 落地(已实现)

让 rollout 采样时用上刚热推的 adapter,链路闭合。

- **`utils/arguments.py`**:加 `--lora-name`(默认 `"policy"`),Block 3 推送与 Block 5 引用共用同一名字。
- **`engine/rollout/sglang_rollout.py`**:`generate()` 构造 payload 时,`lora_enable` 则加 `payload["lora_path"] = args.lora_name`(train + eval 都覆盖)。

两个 sglang 语义要点(读源码确认):
- **generate 的 `lora_path` 字段实际按「已注册 adapter 名」查找**(`tokenizer_manager._resolve_lora_path` → `lora_registry.acquire(lora_path)` → name→lora_id),不是文件路径。Relax 的 from-tensors 加载把 adapter 注册成 `lora_name="policy"`(内部 path 占位 `"__tensor__"`),所以带 `lora_path="policy"` 正好命中。
- **首轮时序无忧**:Relax「训练前必先同步一次权重」(arguments.py `--hf-checkpoint` 注释明示),对 LoRA = 首次 `update_weights` 先推一次 adapter,故首轮 rollout 时 `"policy"` 已在册。
- **驱逐坑已规避**:adapter 被 LRU 驱逐后 sglang 会尝试从 path 重载(我们 path 是假的会失败),但 Block 3 的 `pause→flush→unload→load→continue` 把更新窗口包住、且只有一个 adapter 不触发 LRU,generate 运行时 adapter 始终在册。

### 4. 剩余工作

5 个 Block 全部落地,代码侧无剩余。

| Block | 位置 | 内容 | 状态 |
|---|---|---|---|
| 1 | `backends/sglang/sglang_engine.py` | LoRA 同步/加载/卸载 HTTP 方法 | ✅ 已实现 |
| 2 | `utils/arguments.py` | `--sglang-*` 透传开 LoRA(零代码) | ✅ 已实现 |
| 3 | `weight_conversion/` + `weight_update/update_lora_from_tensor.py` | adapter 改名 + TP/PP 复原 + 打包 + IPC 热推 | ✅ 已实现(§3.7) |
| 4 | `model_provider.py` / `model.py` / `arguments.py` | thinker attn 挂 Megatron-Bridge LoRA | ✅ 已实现(§3.6) |
| 5 | `sglang_rollout.py` / `arguments.py` | rollout payload 加 `lora_path` 引用 `lora_name` | ✅ 已实现(§3.8) |

**真集群端到端联调进展**:① adapter 真生效 → ✅ 已验证（机制层 `verify_lora_learning.py` + 传播层「非零 LoRA 改变输出」）；③ 多步不泄漏/不崩 → ✅ 已验证（`verify_cpu_backup` 2 step colocate RL 跑通、base 存活）；② reward 上升 → ✅ 已验证（翻译 zh→en + BLEU，10 步 BLEU 0.46→0.53 `[PASS]`）。

参考:[Osmosis slime-rl PR#18](https://github.com/Osmosis-AI/slime-rl/pull/18) 做过同类"MoE 模型 + 限定 text decoder + stock sglang 兼容"的处理。

### 5. 复现命令

```powershell
# Windows 控制台需切 UTF-8,否则 modal 打印 ✓ 会崩
chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"

pip install modal
modal token new

modal run modal_run.py                 # 实验 E
modal run modal_run.py --dynamic 1     # 实验 F
modal run modal_run.py::rl             # 实验 G
modal run modal_run.py::flat           # 实验 H
modal run modal_run.py::mega           # 实验 I
modal run modal_run.py::relax          # 实验 J

modal run modal_verify_lora.py         # 验证 K(Megatron-Bridge PEFT 挂 LoRA)

python verify_lora_convert.py          # 验证 L(Block 3 转换器,本地 CPU,零成本)
modal run modal_verify_lora.py::tp     # 验证 M(TP=2 adapter 切分复原,2×T4)
```

### 6. 关键文件

| 文件 | 作用 |
|---|---|
| `modal_run.py` | 全部实验 E–J 的 Modal 驱动 |
| `verify_lora_attach.py` | 验证 K:Megatron-Bridge PEFT 挂 LoRA 的命名/冻结/scoping |
| `modal_verify_lora.py` | 验证 K 的 Modal 驱动(slime 镜像 + T4) |
| `sglang/python/sglang/srt/models/qwen3_omni_moe.py` | `_lora_pattern`(限定 thinker) |
| `sglang/python/sglang/srt/lora/lora_manager.py` | `should_apply_lora` 闸门 |
| `sglang/python/sglang/srt/lora/lora.py` | `normalize_qkv_proj` 等权重规范化(契约源头) |
| `sglang/test/registered/rl/test_lora_load_from_tensor.py` | sglang 官方 RL LoRA 用法模板(含 flattened_bucket) |
| `Relax/relax/backends/sglang/sglang_engine.py` | Block 1:Relax 侧 LoRA 同步方法 |
| `Relax/relax/backends/megatron/model_provider.py` | Block 4:`apply_lora_to_model` / `wrap_model_provider_with_lora` |
| `Relax/relax/backends/megatron/model.py` | Block 4:`setup_model_and_optimizer` 接入 LoRA wrapper |
| `Relax/relax/utils/arguments.py` | Block 4/5:`--lora-*` / `--lora-name` CLI 参数 |
| `Relax/relax/engine/rollout/sglang_rollout.py` | Block 5:`generate()` payload 加 `lora_path` |
| `Relax/relax/backends/megatron/weight_conversion/qwen3_omni_moe.py` | Block 3:adapter→sglang lora 命名转换(qkv 重排 / o_proj) |
| `Relax/relax/backends/megatron/weight_update/hf_weight_iterator_direct.py` | Block 3:`name_filter` 把同步范围缩到 adapter |
| `Relax/relax/backends/megatron/weight_update/update_lora_from_tensor.py` | Block 3:`UpdateLoRAFromTensor` 打包 + IPC 热推 |
| `Relax/relax/backends/megatron/actor.py` | Block 3:`lora_enable` 时切到 `UpdateLoRAFromTensor` |
| `verify_lora_convert.py` | 验证 L:Block 3 转换器纯函数实测(命名/形状/qkv 重排) |
| `verify_lora_tp.py` | 验证 M:TP=2 下 adapter 切分能被 all_gather 正确复原 |

### 7. 结论

sglang 对 RL runtime LoRA 的接收能力(静态/动态/热更新/flattened_bucket/HTTP+IPC)在本地 `main` 上**全部验证通过**;旧版本"加载即崩"的 bug 在 main 已修复;Megatron→sglang 的权重名/形状契约已用 logprob 等价性钉死;Megatron-Bridge PEFT 挂 LoRA 的命名/冻结/scoping 已在真实环境实测(验证 K)。

**5 个 Block 全部落地**:Block 1(sglang 同步方法)、Block 2(参数透传)、Block 3(adapter 改名打包 + IPC 热推,复用全量同步的 TP/PP 复原管线)、Block 4(训练侧挂 LoRA)、Block 5(rollout 引用 adapter)。Block 3 两段核心逻辑分别由 **验证 L**(qkv 重排逐位对)和 **验证 M**(TP>1 切分复原,原唯一残留风险已清零,TP 任意取值都成立)钉死;Block 5 的 sglang `lora_path` 按名查找 + 首轮时序 + LRU 驱逐三个语义点均已读源码确认。

**整条 RL runtime LoRA 链路已打通,代码侧无剩余**。真集群端到端联调:adapter 生效 ✅、多步稳定不泄漏 ✅（`verify_cpu_backup` 2 step）、reward 上升（效果层）✅（翻译+BLEU 10 步 `[PASS]`）。三层次「LoRA 在学」全部验证完毕。

---

## Modal 4×A100 冒烟测试记录(2026-05-31)

> 脚本:`modal_relax_smoke.py` + `Relax/scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh`
>
> 目标:在真实 Relax + 4×A100-80GB + colocate TP4 环境里把 Block 3/4/5 整条 LoRA 链路跑通(不看训练效果,跑 2 个 rollout step 就停)。
>
> 镜像方案:`slimerl/slime:latest` 为底座,叠加 Relax 依赖 + redai fork megatron-bridge,本地 sglang + Relax 源码通过 `add_local_dir` + `PYTHONPATH` 覆盖注入。

### 遇到的坑与修复(按触发顺序)

#### 坑 1:Modal 入口需明确指定

**错误**:`modal run modal_relax_smoke.py` → `Specify a Modal Function or local entrypoint`

**原因**:脚本里同时定义了 `probe` 和 `main` 两个 `@app.local_entrypoint()`，Modal 无法自动选择。

**修复**:显式指定 `modal run modal_relax_smoke.py::main`。

---

#### 坑 2:Windows CRLF 行尾 → Linux bash 报 `$'\r': command not found`

**错误**:shell 脚本每行报 `$'\r': command not found`，脚本整体失败。

**原因**:Windows 下编辑的 `.sh` 文件以 CRLF(`\r\n`)结尾，Linux bash 识别不了 `\r`，把它当命令名。

**修复**:用字节级 PowerShell 脚本把 `Relax/scripts/` 下所有 `.sh` 批量转为 LF：
```powershell
Get-ChildItem "D:\Li_Lab\RL\Relax\scripts" -Recurse -Filter "*.sh" | ForEach-Object {
    $raw = [System.IO.File]::ReadAllBytes($_.FullName)
    $text = [System.Text.Encoding]::UTF8.GetString($raw)
    $fixed = $text -replace "`r`n","`n" -replace "`r","`n"
    $out = [System.Text.UTF8Encoding]::new($false).GetBytes($fixed)
    [System.IO.File]::WriteAllBytes($_.FullName, $out)
}
```

> **注意**:早期用 `Set-Content -Encoding utf8` 的写法会在文件头插入 UTF-8 BOM(`\uFEFF`)，导致 shebang `#!/bin/bash` 变成 `!/bin/bash`（去 BOM 时 `.Substring(1)` 错误截掉了 `#`）。**必须用字节级写法**（`UTF8Encoding(false)` 即 no-BOM），并避免 `Get-Content/Set-Content` 修文本内容。

---

#### 坑 3:source 的 model config `.sh` 也有 CRLF/BOM 损坏

**错误**:`qwen3-omni-30B-A3B.sh: line 1: syntax error near unexpected token 'c'`

**原因**:批量 CRLF 转换脚本的 BOM 处理逻辑用字符串 `.Substring(1)` 删 BOM，实际截掉了 `#`，`# Copyright` 变成 ` Copyright`；bash 把 `Copyright` 误判为命令。

**修复**:`git checkout -- scripts/` 恢复所有被损坏的文件，然后用正确字节级脚本重新转一次（只做换行符替换，不碰内容）。

---

#### 坑 4:`megatron.training` 模块找不到

**错误**:`ModuleNotFoundError: No module named 'megatron.training'`

**原因**:Megatron-LM 在 slime 镜像里装在 `/root/Megatron-LM`（editable install），但 ray job 的 `runtime-env-json` 里的 `PYTHONPATH` 只设了 sglang 和 Relax，没有 Megatron-LM。Ray 启动 worker 时用的是新的 runtime env，不继承宿主 `site-packages`（里 editable 的 `.pth` 文件）。

**修复**:在 `modal_relax_smoke.py` 的 `MEGATRON_REMOTE` 常量和 `PYTHONPATH` 里加上 `/root/Megatron-LM`：
```python
MEGATRON_REMOTE = "/root/Megatron-LM"
PYTHONPATH = f"{SGLANG_REMOTE}:{RELAX_REMOTE}:{MEGATRON_REMOTE}"
```

---

#### 坑 5:`args.norm_epsilon` 属性不存在

**错误**:`AttributeError: 'Namespace' object has no attribute 'norm_epsilon'`

**位置**:`relax/backends/megatron/arguments.py:_hf_validate_args`

**原因**:Relax 在校验 HF config 与 Megatron args 是否一致时，映射表写的是 `"norm_epsilon"`，但 Megatron 最新版的 CLI 参数 `--norm-epsilon` 实际存储在 `args.layernorm_epsilon`（定义在 `megatron/core/transformer/transformer_config.py`：`layernorm_epsilon: float = field(default=1e-5, metadata={"argparse_meta": {"arg_names": ["--norm-epsilon"]}})`）。

**修复**:
```python
# 改前
("rms_norm_eps", "norm_epsilon", equal),
# 改后
("rms_norm_eps", "layernorm_epsilon", equal),
```

---

#### 坑 6:`--optimizer-cpu-offload` 需要配套 `--use-precision-aware-optimizer`

**错误**:
```
AssertionError: The optimizer cpu offload must be used in conjunction with
`--use-precision-aware-optimizer`, as the hybrid device optimizer reuses
the code path of this flag.
```

**位置**:`Megatron-LM/megatron/training/arguments.py:1190`

**原因**:Megatron 新版本要求 CPU offload 优化器必须和 precision-aware 优化器联用，该 assert 是新增的强制检查。冒烟脚本 `OPTIMIZER_ARGS` 里只有 `--optimizer-cpu-offload`，缺了后者。

**修复**:在冒烟脚本 `OPTIMIZER_ARGS` 里加上 `--use-precision-aware-optimizer`：
```bash
OPTIMIZER_ARGS=(
   --optimizer adam
   ...
   --optimizer-cpu-offload
   --use-precision-aware-optimizer   # ← 新增
)
```

---

#### 坑 7:Qwen3-Omni 的 chat_template 在 processor 不在 tokenizer（**已实查证实，非臆断**）

**错误**:
```
ValueError: Cannot use chat template functions because tokenizer.chat_template is
not set and no template argument was passed!
```

**走过的弯路**:第一反应是"去掉 `--apply-chat-template`"。但去掉后又撞另一个断言：
```
AssertionError: prompt must be a list when processor is not None, got <class 'str'> instead
```
因为 Qwen3-Omni 强制带 processor，`process_raw_sample` 里 `if processor:` 分支强制要求 prompt 为对话 list（`data_utils.py:239`），绕不过去。说明**对 Qwen3-Omni 必须用 chat template**。

**实查 ground truth**（`modal run modal_relax_smoke.py::inspect`，纯 CPU 几分钱）。Volume 上的 checkpoint：
| 文件 | chat_template? |
|---|---|
| `tokenizer_config.json` | ❌ 不含 |
| `chat_template.json` | ✅ 含（官方模板，6519 字符）|
| `AutoTokenizer.chat_template` | ❌ None |
| `AutoProcessor.chat_template` | ✅ 存在 |

**根因**:Qwen3-Omni 的 chat_template 放在 **processor**（`chat_template.json`）里，**不在** tokenizer。而 Relax 的 `process_raw_sample` 调的是 `tokenizer.apply_chat_template`（`data_utils.py:225`）—— 对这种"模板仅存于 processor"的 checkpoint 就拿不到模板。这是 **Relax 的潜在 bug**（官方 16 卡 omni 脚本同样用 `tokenizer.apply_chat_template`，若在同版本 transformers + 此 checkpoint 上跑也会中招；推测官方此前用的 checkpoint 把模板冗余写进了 tokenizer_config，或 transformers 旧版会把 chat_template.json 读进 tokenizer）。

**修复**(冒烟侧，不动 Relax 框架):保留 `--apply-chat-template`，把 checkpoint 自带的**真实**模板经 `--apply-chat-template-kwargs '{"chat_template": "..."}'` 显式喂进去。Modal wrapper 的 `_chat_template_kwargs()` 优先读 `chat_template.json`（命中真实 6519 字符模板），缺失才退回内置 ChatML 兜底。脚本侧：
```bash
if [ -n "${CHAT_TEMPLATE_KWARGS:-}" ]; then
   ROLLOUT_ARGS+=(--apply-chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}")
fi
```

> **若要从根上修 Relax**:应让 `load_tokenizer` 在 tokenizer 无 chat_template 时回落到 processor 的 chat_template（或数据处理直接用 `processor.apply_chat_template`）。本次冒烟先用注入法绕过，不扩大改动面。

---

#### 坑 8/9:LoRA wrapper 破坏 Megatron 的「按签名派发参数」，撞 `unexpected keyword argument 'config'`（**真 bug，非环境**）

**错误**:
```
TypeError: ... wrapped_provider() got an unexpected keyword argument 'config'   （坑 8）
TypeError: Qwen3OmniModelProvider.provide() got an unexpected keyword argument 'config'   （坑 9）
```
actor 在 `MegatronTrainRayActor.init()` 里反复崩溃 → Ray Serve 不停重启 replica，白烧 4×A100。

**根因（两段式，关键）**:Megatron-LM 的 `build_model` 会**按 provider 的真实签名**决定传哪些 kwarg（`config`/`pg_collection`）。bridge 模式下 `get_model_provider_func` 返回的是 `ModelProvider.provide`（`model_provider.py:165`），其签名只有 `(pre_process, post_process, vp_stage)`，**不收** `config`。
- **坑 8**：最初 wrapper 把签名写死成 `(pre_process, post_process, vp_stage)`，Megatron 用 `config=` 调它 → wrapper 自己报 unexpected kwarg。
- **坑 9**（改错方向后暴露）：把 wrapper 改成 `(*args, **kwargs)` 后，签名探测**误以为它"什么都收"**，于是 Megatron 把 `config=` 传进来，wrapper 再原样转发给只认三参的 `provide` → 这次换 `provide` 自己报 unexpected kwarg。非 LoRA 路径之所以没事：裸 `provide` 被直接 inspect，签名里没 `config`，Megatron 就不传。

**正确修复**(`model_provider.py`):wrapper 必须**透传原 provider 的签名**（让 Megatron 探测看到与裸 provider 一致的参数），并在运行期按原签名**过滤多余 kwarg**（兼容"无条件传 config"的 Megatron 版本）：
```python
@functools.wraps(original_provider)          # 让 inspect.signature 透出原签名
def wrapped_provider(*provider_args, **provider_kwargs):
    if allowed_keys is not None and not accepts_var_kw:
        provider_kwargs = {k: v for k, v in provider_kwargs.items() if k in allowed_keys}
    model = original_provider(*provider_args, **provider_kwargs)
    apply_lora_to_model(model, args)
    return model
```

> 该修复已被 **Tier 0 签名自测**（`modal ... ::wiring`，纯 CPU 秒级）和 **Tier 1 端到端**（`::tier1`，T4）双重验证：用带 `config=/pg_collection=` 的真实调用方式打过去不再崩。

---

#### 坑 10:`from megatron.bridge.peft import LoRA` 在 redai fork 里 ImportError（**Tier 1 在 T4 上抓到，省下一轮 A100**）

**错误**:
```
ImportError: cannot import name 'LoRA' from 'megatron.bridge.peft'. Did you mean: 'lora'?
```

**原因**:redai fork 的 `LoRA` 在子模块 `megatron.bridge.peft.lora`，顶层 `peft/__init__.py` 没有 re-export。`apply_lora_to_model`（`model_provider.py:321`）用的是顶层导入，必崩——而这步发生在 actor 真正挂 LoRA 时，等同于又一个"只有上 GPU 建模型才触发"的坑。

**修复**(`model_provider.py`):优先从子模块导入，失败再回退顶层：
```python
try:
    from megatron.bridge.peft.lora import LoRA
except ImportError:
    from megatron.bridge.peft import LoRA
```
> `verify_lora_attach.py` 早有 `_import_lora()` 的兼容写法，但生产代码漏了同步——Tier 1 的价值正在于把这种"测试脚本对、生产代码错"的偏差在 T4 上暴露。

---

#### 坑 11:bridge 加载权重报 `Cannot determine parallelism type for 'LinearCrossEntropyModule'`（**环境/版本，非 LoRA**）

**错误**（Tier 2 真冒烟，LoRA 已成功挂上**之后**、加载 30B 权重时）:
```
ValueError: Cannot determine parallelism type for module 'LinearCrossEntropyModule'
at weight 'thinker.language_model.output_layer.weight'.
```

**原因**:Qwen3-Omni 的 `output_layer` 是 `LinearCrossEntropyModule`（bridge 模型定义里省显存用）。Megatron-Bridge 的 `AutoMapping` 用一张 `_MODULE_TYPE_REGISTRY` 判断每个权重的并行切分方式；镜像里 force-install 的 redai fork pin 提交 `f13bec09` **较老、注册表里还没收录** 这个类型（本地新版 bridge 已把它登记为 `"column"`）。与 LoRA 无关——非 LoRA bridge 路径同样会中招。

**修复**(`model_provider.py` bridge 分支，幂等补注册)：
```python
from megatron.bridge.models.conversion.param_mapping import AutoMapping
AutoMapping.register_module_type("LinearCrossEntropyModule", "column")
```
> 这是 class 级全局注册表，每个 rank 进程在 `get_model_provider_func` 里注册一次即对后续 actor/ref 权重加载全程生效。值取 `"column"` 与新版 bridge 默认一致，确定无误。

> **真冒烟里程碑**：本轮在真实 30B 上看到 PEFT 逐层打印 `Adding lora to: thinker.language_model.decoder.layers.{N}.self_attention.{linear_qkv,linear_proj}`，随后 `[LoRA] adapter 已挂载：可训参数 192 个 / 2.556M 元素，冻结 4287 个`——证明 scoped target 在**真实 Qwen3-Omni** 上精确命中文本侧、未误挂 audio/vision，且只有 adapter 可训（Part B 因 OOM 没拿到的真实确认，这里补上了）。修复后权重加载（`Load checkpoint from HuggingFace model into Megatron`）顺利越过该步。

---

#### 坑 12:sglang 起服务报 `flashinfer_python 0.6.3 < 0.6.11.post1`（**本地 sglang 覆盖导致的版本错配**）

**错误**（actor/权重都就绪后，rollout 起 sglang server 时）:
```
Exception: flashinfer_python is installed with version 0.6.3, which is less than
the minimum required version 0.6.11.post1.
（sglang_engine: "Server process terminated unexpectedly."）
```

**原因**:我们用 PYTHONPATH 把镜像自带 sglang 覆盖成了**本地这份较新的 sglang**（含 Block 1/2 的 Qwen3-Omni LoRA 改动），它在 `_set_envs_and_config` 里断言 `flashinfer_python>=0.6.11.post1`；但 slime 镜像里是 0.6.3（镜像原配旧 sglang 本来匹配，是覆盖造成的错配）。该断言**仅在 `attention_backend=="flashinfer"`**（默认）时触发；其后还有一条不分 backend 的 `sgl-kernel>=0.4.2.post2` 断言。

**修复**(冒烟侧，不动 sglang 源码)：
- 脚本 `SGLANG_ARGS` 加 `--sglang-attention-backend triton`——绕开 flashinfer 断言，且运行时不实际调用旧 flashinfer（冒烟只验 LoRA 链路，不依赖 flashinfer attention）。
- env + Ray runtime env 加 `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`——跳过 sgl-kernel 版本断言（probe 已确认本地 sglang 能 import、sgl-kernel 二进制兼容）。

> **根上的解法**（非冒烟）：让镜像的 flashinfer/sgl-kernel 与本地 sglang 版本对齐（升级 flashinfer 到 ≥0.6.11.post1），或用与本地 sglang 配套的基础镜像。

---

#### 坑 13:sglang 在 cuda graph 捕获时崩 `custom_all_reduce: CUDA error: invalid argument`（**同源版本错配**）

**错误**（sglang server 加载完模型、分配 KV cache、`Capturing batches` 到 100% 后）:
```
[TPx] Scheduler hit an exception: ... cuda_graph_runner.capture()
  → custom_all_reduce_v2.py: share_graph_inputs()
RuntimeError: CUDA error: invalid argument (custom_all_reduce.cuh:37)
```

**原因**:与坑 12 同源——本地注入的较新 sglang 的 **custom all-reduce v2** 内核（`tvm_ffi`/`jit_kernel`）与镜像里的 sgl-kernel 二进制不匹配，TP=4 在 cuda graph 捕获阶段对 TP all-reduce 做 `share_graph_inputs` 时报 invalid argument。

**修复**(脚本 `SGLANG_ARGS`)：
```bash
--sglang-disable-cuda-graph          # 跳过 graph 捕获（崩溃发生处）
--sglang-disable-custom-all-reduce   # TP all-reduce 退回 NCCL，绕开不匹配的自定义内核
```
> 冒烟只验 LoRA 链路，这两项是纯性能优化，关掉不影响正确性。根上仍是版本对齐。

---

### 分档验证策略（先便宜后烧卡）

把"真烧 4×A100 才暴露"的接线 bug 下沉到秒级/分钟级的便宜档：

| 档 | 命令 | 资源/耗时 | 覆盖 | 能提前抓到的坑 |
|---|---|---|---|---|
| Tier 0 | `modal run modal_relax_smoke.py::wiring` | CPU / 秒级 | provider 包装层签名派发 | 坑 8/9 |
| Tier 0 | `modal run modal_relax_smoke.py::inspect` | CPU / 秒级 | tokenizer/processor chat_template 实查 | 坑 7 |
| Tier 1 | `modal run modal_relax_smoke.py::tier1` | T4 / 几分钟 | 真实 Megatron 模块走 wrap→attach→convert 全链 + 真实 bridge 识别 tiny config | 坑 8/9/10 + target 匹配/命名契约 |
| Tier 2 | `modal run modal_relax_smoke.py` | 4×A100 | 真权重 + 真 rollout/同步 | （仅在 0/1 全绿后跑一次）|

**Tier 1 脚本** `verify_lora_e2e.py`（也可本地 `torchrun --nproc_per_node=1 verify_lora_e2e.py`）：
- **Part A（必跑，已全绿）**:手搭真实 Megatron self-attention 组成的 `thinker.{language_model, audio_model}` 树 → `wrap_model_provider_with_lora` → 按 Megatron `build_model` 方式带 `config=/pg_collection=` 调用 → `_assert_lora_attached` → 对每个 adapter 跑 `convert_qwen3omni_to_hf`，断言落到 `base_model.model.thinker.model.layers.N.self_attn.{qkv_proj,o_proj}.{lora_A,lora_B}`，四件齐全、未误挂多模态塔。
- **Part B（尽力，bonus）**:读 Volume 真 config，缩层数/专家数/vocab 后用真实 `AutoBridge.from_hf_pretrained` 构造 → 已确认 redai bridge **识别 Qwen3-Omni 并建出 `Qwen3OmniModelProvider`**；全家桶（thinker+audio+vision+talker）的**完整实例化**受显存/meta 限制（T4/L4 OOM、meta 张量不可 copy），非阻断 SKIP——实际挂载语义已由 Part A 在真实 Megatron 模块上覆盖。

---

### 冒烟进展状态(截至记录)

| 阶段 | 状态 |
|---|---|
| Modal 镜像构建 | ✅ |
| Ray head 启动 | ✅ |
| 参数解析(`parse_args`) | ✅（修复坑 4/5/6 后通过）|
| Ray 集群初始化 + DCSCoordinator | ✅ |
| GPU 0-3 资源分配 | ✅ |
| chat_template 注入（坑 7） | ✅（用 checkpoint 自带真实模板）|
| Tier 0 签名自测（坑 8/9） | ✅ 纯 CPU 通过 |
| Tier 1 端到端 wrap→attach→convert（坑 8/9/10） | ✅ T4 Part A 全绿 |
| SGLang 引擎建模/KV 分配/`fired up`（坑 11/12/13） | ✅ Tier 2 已越过，server 健康 `health_generate 200` |
| sgl-router 注册 tokenizer（坑 14） | ✅ 生成 `tokenizer.json` 后通过 |
| 首个 rollout + 训练步 + 进入 `update_weights` | ✅ 已跑到 LoRA 同步路径 |
| LoRA adapter 命名映射同步（坑 15） | ✅ Tier 1（T4）命名回归通过 |
| 进入 `update_weights` 第一步 TP gather（坑 16） | 🟡 回退 direct + contiguity 修复；Tier 1 接线绿，待 4×A100 验证 |

#### 坑 17:镜像 pin 的 redai-fork bridge 没有 `export_adapter_weights`
- **现象**：切到 Bridge 路线后，Tier 2 在 `update_weights` 报 `AttributeError: 'AutoBridge' object has no attribute 'export_adapter_weights'`。
- **根因**：镜像 `pip install git+...redai-infra/megatron-bridge@f13bec09`，该提交**早于** adapter 导出特性（上游 5 月才把 `export_adapter_weights`/`stream_adapter_weights_megatron_to_hf` 加进 peft_bridge）。redai fork 的 `f13bec09` 之所以被 pin，是因为它带 **Qwen3-Omni** 支持，而上游主线只有 Qwen2.5-Omni——升级 bridge 很可能丢掉 Qwen3-Omni，且要重建镜像（贵、风险高）。
- **决策**：放弃 Bridge export 路线，**回退 direct（`HfWeightIteratorDirect` + `convert_qwen3omni_to_hf`）**，并正面修复坑 16 的 TP gather。
- **坑 16 根因（再判一）**：~~adapter 是连续 buffer 的非连续视图，all_gather 前需 `.contiguous()`~~ —— 加了 contiguity 后**仍复现**，证伪。
- **坑 16 根因（再判二，当前）**：全量路径用 `weights_backuper.get("actor")`（`translate_gpu_to_cpu` 的 CPU 常驻副本）能跑通，LoRA 路径(坑 15)改成直接读 **live 参数**才崩 —— 唯一差异即 getter 张量来源。colocate 下 `update_weights` 时训练模型被 `torch_memory_saver` offload，live GPU 显存已释放，`all_gather` 读其 `.data` = 野指针 → illegal access。
- **修复（待验证）**：LoRA getter 同加 `translate_gpu_to_cpu=True`（offload 时取 CPU backup，未 offload 时退回 live，均安全）；contiguity 保留为防御；并在 param_info 跨 rank 校验里加 attrs(TP 属性)一致性断言，把潜在的集合通信错配转成清晰报错。
- **诊断**：`update_weights` 在 rank0 打印取到的 adapter 张量 device/contiguous，确保下次运行可定位。

#### 坑 16:TP=4 下手写 adapter all-gather 触发 CUDA illegal memory access
- **现象**：坑15 修复后 30B 真冒烟一路跑到 `update_weights`（日志 `[LoRA] adapter 已挂载：可训参数 192 个`、sglang `fired up`、`Update weights: 0/1`），随即 rank2/rank3 NCCL watchdog 报 `CUDA error: an illegal memory access was encountered`，发生在 `get_hf_weight_chunks` 第一次迭代的 `_get_megatron_full_params`/`all_gather_params_async`（手写 TP all-gather adapter 分片）。TP=1 的 Tier 1 测不到，TP=4 才暴露。
- **根因方向**：手写 direct gather 对 Megatron-Bridge PEFT 的 `ParallelLinearAdapter` 分片（linear_in/linear_out 的 TP 属性、可能的非连续 DDP buffer view）处理不稳。这正是 Miles **显式废弃**手写 `convert_lora_to_hf()`、改用 `bridge.export_adapter_weights()` 的原因。
- **决策**：放弃手写 raw/direct gather+convert，改走 **bridge `export_adapter_weights()`**——已确认 redai 版 `AutoBridge` 提供该 API（`peft_bridge.stream_adapter_weights_megatron_to_hf`），文档原话「Export only adapter weights **without merging** into base tensors」，**内部处理 TP/PP/EP gather + fused-qkv 拆分**，同时满足「只 lora、无 base、无 merge」的硬约束。
- **改动（已落地）**：
  1. `HfWeightIteratorBridge.get_hf_weight_chunks(.., weight_type="base"|"lora")` 加 lora 分支：`export_adapter_weights(self.model, cpu=False)` + 去 `.base_layer.` + 过滤 `is_lora_weight_name`（`.lora_A./.lora_B.`），lora 不量化（镜像 Miles）；
  2. `UpdateLoRAFromTensor` 改用 `HfWeightIteratorBridge`（不再 `HfWeightIteratorDirect`），以 `weight_type="lora"` 调用；adapter 由 Bridge 直接从模型读，故 `actor.py` 的 `weights_getter` 退化为 `lambda: {}`；
  3. 手写 `convert_qwen3omni_to_hf` 的 LoRA 段 + `_reorder_qkv_lora_b` 不再用于同步（仍保留供 base raw 模式）。
- **sglang 兼容性已核实**：你这份 sglang `lora.py::normalize_qkv_proj` **两种命名都吃**——分离 `q/k/v_proj`（Bridge 标准）会被自动 stack 成 `qkv_proj`；按 `get_layer_id`+模块子串匹配，`thinker.`/`base_model.model.` 前缀不影响。
- **Tier 1 接线自检**：`verify_lora_e2e.py::_run_sync_wiring_check` 断言 `UpdateLoRAFromTensor` 已用 Bridge、不再引用 Direct、以 `weight_type='lora'` 调用、`is_lora_weight_name` 判定正确。

#### 坑 15:bridge 模式下 LoRA 同步的命名不一致 → `update_weights` KeyError
- **现象**：tokenizer 修复后一路跑到 `MegatronTrainRayActor.update_weights()`（说明建模型、挂 LoRA、rollout、训练步全过了），在 `update_lora_from_tensor.py:161` 报
  `KeyError: 'module.module.thinker.language_model.decoder.layers.0.self_attention.linear_proj.adapter.linear_in.weight'`（按名排序后第一个 adapter，并非 linear_proj 特有）。
- **根因**：
  - 全量同步走工厂 `HfWeightIteratorBase.create`，在 `--megatron-to-hf-mode bridge` 下选 `HfWeightIteratorBridge`，且 `weights_backuper` 的 source 用 `convert_to_global_name=(mode=="raw")=False` → **vanilla 命名** `vp_stages.N.<stripped>`。
  - 但我把 `UpdateLoRAFromTensor` 写死成 `HfWeightIteratorDirect`（raw/direct），其 `param_infos` 用默认 `convert_to_global_name=True` → **全局命名** `module.module.thinker...`，转换器 `convert_qwen3omni_to_hf` 也是面向全局命名、且 LoRA-aware（认 `.adapter.`）。
  - 迭代器拿「全局名」去 backuper 的「vanilla-name dict」里 `megatron_local_weights[info.name]` → 必然 KeyError。
- **为何不直接改用 bridge iterator**：megatron-bridge 的 `get_conversion_tasks/export_hf_weights` 只认基座 HF 结构，**不认 LoRA adapter**（无法把 `adapter.linear_in/out` 导成 `lora_A/lora_B`）。所以 LoRA 同步必须走 LoRA-aware 的 raw 转换器。
- **修复**：保留 raw/direct + `convert_qwen3omni_to_hf`，只把 `actor.py` LoRA 分支的 `weights_getter` 从 `weights_backuper.get("actor")`（bridge 模式 vanilla 命名）改成 `dict(named_params_and_buffers(args, model, convert_to_global_name=True))`——直接取**全局命名的实时 adapter 参数**，与迭代器 `param_infos` 命名严格一致。adapter 权重很小，实时读取无开销。
- **最小成本验证（不烧 4×A100）**：在 `verify_lora_e2e.py` Part A 加 `_run_sync_naming_regression`，用手搭的真实 Megatron 树跑**完全相同**的代码路径——`dict(named_params_and_buffers(..., convert_to_global_name=True))` 作 weights_getter、`HfWeightIteratorDirect(name_filter=_is_lora_adapter_name)` 建 `param_infos`，逐个复现当时 KeyError 的 `weights[info.name]` 查表。Tier 1（T4，约 4 分钟，exit 0）结果：weights_getter 产出 4 个 adapter 键、迭代器收集 4 个 param_info、**名字全部命中、缺失=[]**，坑 15 不再复现。

#### 坑 14:sgl-router 需要 fast `tokenizer.json`，Qwen3-Omni 只给了 slow BPE
- **现象**：sglang 引擎已 `The server is fired up` 且 `GET /health_generate 200 OK`，但 Relax 用的 **sgl-router（Rust）** 报
  `tokenizer_registration: Failed to load tokenizer '/models/qwen3-omni': Directory does not contain a valid tokenizer file (tokenizer.json, tokenizer_config.json, or vocab.json)` → RolloutManager 初始化失败 → 整个 app `SystemExit(1)` 退出。
- **根因**：Qwen3-Omni checkpoint 只带 `vocab.json + merges.txt`（slow BPE），**没有 `tokenizer.json`**；sgl-router 的 Rust `tokenizers` crate 必须吃 HF fast 格式的 `tokenizer.json`，对 vocab.json+merges.txt 不买账。与 LoRA 无关，是数据/环境缺件。
- **修复**：在便宜的 CPU 函数 `inspect_ckpt`（及 `smoke` 兜底）里用 `AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True).save_pretrained(MODEL_DIR)` 落一份 `tokenizer.json`，并 `model_volume.commit()` 持久化到 Volume（幂等，存在即跳过）。这样 A100 那次直接跳过、不浪费机时。

> **下一步**:先跑 `modal run modal_relax_smoke.py::inspect`（CPU，秒级）生成并 commit `tokenizer.json`，再重跑 Tier 2 真冒烟（`modal run modal_relax_smoke.py`），观察 sgl-router 注册成功、首个 rollout 与 `UpdateLoRA` 同步。

---

### 坑 16 最终定位：Tiny Qwen3-Omni TP=4 全链路验证（2026-05-31）

#### 验证方案
用**保留真实 Qwen3-Omni 架构特征（GQA 32Q/4KV、MoE、fused QKV）但只有 2 层 4 专家**的超小模型，在 4x T4（~$0.15/次）上跑 TP=4 真实 NCCL all_gather。

缩容配置（关键结构保持原值）：
- `num_hidden_layers=2`（48→2）, `num_experts=4`（128→4）, `vocab_size=4096`
- `hidden_size=2048`, `num_attention_heads=32`, `num_key_value_heads=4`, `head_dim=128`

GQA 关键维度：
- `Q_OUT = 32×128 = 4096`（≠ hidden=2048）
- `QKV_OUT = (32+4+4)×128 = 5120`

#### 对比的三种 Gather 方法
| 方法 | 结果 |
|------|------|
| A. Relax `all_gather_params_async`（用 `param.partition_dim`） | ALL PASS |
| B. Slime `_smart_gather`（推断 expected_full_shape 找 shard axis） | ALL PASS |
| C. Bridge `export_adapter_weights`（官方 API） | 镜像版本过旧，不可用 |

#### 验证结果（全部通过）
```
[ALL PASS] Stage 1-3: 8 参数 × 2 方法，全部形状/数值一致

GQA 关键断言：
  linear_proj.adapter.linear_in → (16, 4096) 正确（rank, Q_OUT）
  linear_qkv.adapter.linear_out → (5120, 16) 正确（QKV_OUT, rank）

非连续 buffer 测试：
  所有 4 个参数非连续 gather 均成功（.contiguous() 修复有效）
```

#### 结论
1. **TP gather 逻辑本身是正确的**：Relax 和 Slime 的 gather 在 GQA Qwen3-Omni 架构上均能正确收集分片 adapter 参数，形状和数值完全一致。
2. **`.contiguous()` 修复对非连续 buffer 有效**：模拟 distributed optimizer 创建的非连续视图，加 `.contiguous()` 后 NCCL all_gather 不再崩。
3. **坑 16 在全 Relax 中的真正根因锁定为 `torch_memory_saver` GPU 内存释放**：最小链路验证表明 gather 逻辑和非连续修复都是正确的，剩余嫌疑人只有 `torch_memory_saver` 将参数底层 GPU storage 释放（offload 到 CPU），导致 `.data` 指向已释放显存。已应用的 `translate_gpu_to_cpu=True` 修复针对此场景。
4. **Stage 4 (SGLang 热加载)**: Server 启动成功但内部 JIT kernel 对合成 tiny 词表有边界 bug（`resolve_future_token_ids` 要求 token 连续性），与 LoRA 无关。

#### 验证脚本
- `verify_tp_gather.py`：Modal 4x T4，~$0.15/次，含 torchrun TP=4 gather + SGLang 热加载
- 运行：`modal run verify_tp_gather.py::main`

---

### 坑 18–21：4xA100 Full Smoke 连续修复（2026-06-01）

在 T4 最小验证确认 gather 正确后，转入 4xA100 Full Smoke（`modal run modal_relax_smoke.py::smoke`），逐步打通 LoRA 全链路。

#### 修复清单

| 坑 | 错误现象 | 根因 | 修复 | 文件 |
|---|---|---|---|---|
| 18 | `AttributeError: '_rebuild_cuda_tensor_original'` | SGLang `tp_worker.load_lora_adapter_from_tensors` 反序列化前未调 monkey_patch | 添加 `monkey_patch_torch_reductions()` 调用 | `sglang/srt/managers/tp_worker.py:193` |
| 19 | `400 Bad Request: 'peft_type'` KeyError | `config_dict` 缺少 `peft_type` 字段 | 添加 `"peft_type": "LORA"` | `Relax/.../update_lora_from_tensor.py:81` |
| 20 | TP0 成功但 TP2/3 `CUDA error: invalid argument` + `AuthenticationError: digest rejected` | ForkingPickler 对 GPU tensor 用 CUDA IPC（跨 TP 不行），对 CPU tensor 用 fd sharing（跨 Ray actor 不行） | 改用 `pickle.dumps` + base64（数据 inline，无 IPC） | `Relax/.../update_lora_from_tensor.py:209` |
| 20b | `IndexError: tuple index out of range` in `_reduce_tensor_modified` | `monkey_patch` 的 `reduce_tensor` 对 CPU tensor tuple 尝试修改第 6 个元素（仅 CUDA 有） | 添加 `len(output_args) > index` 守卫 | `sglang/srt/utils/patch_torch.py:74` |
| 21 | `TypeError: ...forward() got unexpected keyword argument 'packed_seq'` | Megatron-LM GPT model 传 `packed_seq`/`cp_group` 给 RoPE，Qwen3-Omni Bridge 不接受 | `patch_rotary_embedding` 只传位置参数 `_original_forward(self, *args)` | `Relax/relax/backends/megatron/__init__.py:30` |

#### 关键证据（第 7 次运行，全流程通过）

```
[LoRA-sync][rank0] 取到 adapter 张量 192 个; device=cpu contiguous=True
Update weights: 100%|██████████| 1/1 [00:00<00:00, 1.50it/s]

Start load Lora adapter from tensors. Lora name=policy
POST /load_lora_adapter_from_tensors HTTP/1.1" 200 OK
TP0: LoRA adapter loading from tensors completes
TP1: LoRA adapter loading from tensors completes
TP2: LoRA adapter loading from tensors completes
TP3: LoRA adapter loading from tensors completes

LoRA adapter '...': loaded weights for target modules ['o_proj', 'qkv_proj'].  (×4 TP)

Prefill batch, #new-seq: 31, #new-token: 1728, token usage: 0.00, input throughput: 688 tok/s
Decode batch, #running-req: 32, gen throughput: 308 tok/s
Rollout generation: 100%|██████████| 32/32 [00:48<00:00, 1.52s/it]

step 0: {'train/loss': 0.0, 'train/pg_loss': 0.0, 'train/entropy_loss': 9.197, 'train/pg_clipfrac': 0.0, 'train/ppo_kl': 0.0}
step 1: {'train/loss': 0.0, 'train/pg_loss': 0.0, 'train/entropy_loss': 9.188, 'train/pg_clipfrac': 0.0, 'train/ppo_kl': 0.0}

Job 'raysubmit_CArC16X5dL95UDxE' succeeded
[exit] smoke returncode=0
```

#### 序列化方案对比

| 方案 | 发送侧 | 接收侧 | 能跨 Ray actor? | 能跨 TP? |
|------|--------|--------|:-:|:-:|
| ForkingPickler + GPU tensor | CUDA IPC handle | rebuild_cuda_tensor | ✓ (同 GPU) | ✗ |
| ForkingPickler + CPU tensor | Unix fd sharing | rebuild_storage_fd | ✗ (authkey 不同) | — |
| **pickle.dumps + CPU tensor** | 数据 inline bytes | SafeUnpickler | **✓** | **✓** |

#### 结论

1. **LoRA RL 全链路（①→⑦）全部打通**：从 Megatron LoRA 训练 → gather → HF 转换 → 序列化 → SGLang 4 TP 热加载 → rollout 推理 → 训练 forward/backward → 多步 GRPO 训练，全流程 0 错误完成。
2. **坑 16 的 `translate_gpu_to_cpu=True` 在真实 30B 模型上验证有效**（adapter 从 CPU 取到、gather 无 CUDA error）。
3. **训练指标合理**：初始 LoRA 参数下 loss=0、pg_clipfrac=0（policy 与 ref 完全一致，优势为零）；entropy_loss ~9.19（词表大小对数级别，符合预期）。
4. **性能参考**：4×A100-80GB colocate 模式，rollout 32 samples 耗时 48s（含首批 prefill），inference throughput 308 tok/s。

---

### 坑 22：SGLang rollout 输出均匀分布（LoRA 加载后模型失效）

| 现象 | 诊断 | 怀疑根因 | 状态 |
|------|------|----------|------|
| rollout 生成全是乱码，`rollout_log_probs = -11.93 ≈ -ln(152064)` = 完全均匀分布；所有 sample 打满 max_response_len；reward 全 0；pg_loss=0（无学习信号） | 见下方分层验证 | SGLang LoRA 对 packed qkv_proj 的 TP shard 逻辑不正确 | 🔍 定位中 |

#### 分层验证（隔离变量）

| 层 | 测试方式 | 结果 |
|----|----------|------|
| Megatron 训练侧 | 观察 `entropy_loss=10.79`（< ln(vocab)=11.93） | ✅ 模型正常工作 |
| SGLang base model（无 LoRA） | `modal_relax_smoke.py::standalone`，不加 LoRA，直接 generate | ✅ 正确回答 "2+3=5"、"capital of France is Paris" |
| SGLang + LoRA（zero-init B） | 需验证（Modal GPU quota 受限） | ❓ 待确认 |
| Relax 全链路 (learn) | 50 step GRPO，观察 metrics | ❌ 输出均匀分布 |

#### 关键证据

- **Base model 正常**（standalone 无 LoRA 生成正确），排除模型权重加载 / 架构 / attention backend 问题
- **LoRA 初始化 B=0，理论上应无影响**：`output = base(x) + α/r × 0 × A(x) = base(x)`
- **但实际加载 LoRA 后输出变为均匀分布**：说明 SGLang 的 LoRA 应用逻辑对 Qwen3-Omni 有 bug

#### 代码分析结果：SGLang LoRA TP 逻辑对 Qwen3-Omni GQA 是正确的

深入追踪后发现 SGLang 的 LoRA 加载链路 **逻辑上正确**：

**Relax 推送流程** (`update_lora_from_tensor.py`):
1. Megatron TP all_gather → 得到 FULL adapter 权重
2. `_convert_qwen3omni_lora_adapter` 转换命名：`linear_qkv.adapter.linear_out → qkv_proj.lora_B`（经 `_reorder_qkv_lora_b` 重排为 [q;k;v] 拼接）
3. `linear_qkv.adapter.linear_in → qkv_proj.lora_A`（单个共享矩阵）
4. 推送 FULL 权重到 SGLang

**SGLang 接收流程** (`lora.py → mem_pool.py → layers.py`):
1. `normalize_qkv_proj`：检测到 `"qkv_proj"` → `lora_A.repeat(3,1)` 得 `[3r, hidden]`；B 无操作保持 `[5120, r]`
2. `slice_lora_b_weights` (`QKVParallelLinearWithLoRA`)：
   - `q_proj_shard_size=1024`, `kv_proj_shard_size=128`, `num_kv_head_replicas=1`
   - 正确切出 per-rank B: `[1024+128+128, r] = [1280, r]`
3. `slice_lora_a_weights`：无切片（column-parallel 输入是 full hidden_size）
4. Buffer 分配 `_column_parallel_lora_b_per_rank_dim`：检测到 `num_kv_heads(4) >= tp_size(4)` → `divide(5120,4)=1280` ✓

**结论**：TP shard 维度对齐，不是 TP 逻辑 bug。

#### 真正怀疑的根因（缩小范围）

既然 LoRA TP 逻辑正确，问题可能在于：

| 假说 | 证据 | 待验 |
|------|------|------|
| A. `enable_memory_saver` 恢复 base 权重时 LoRA-wrapped module 出错 | 仅 colocate 模式下观察到，standalone base 正常 | ❓ 需要 standalone + LoRA 测试（被 GPU quota 终止） |
| B. CUDA graph 与 LoRA forward 路径不兼容 | Relax 使用 `disable_cuda_graph=True` → **排除** | ✗ |
| C. Adapter 命名前缀不匹配 `get_target_module_name` | 代码确认 normalized target_modules={"qkv_proj","o_proj"}，名字匹配 | ✗ |
| D. 初次 rollout 时 SGLang 还未收到 adapter（时序问题） | smoke 测试 2 step 成功，learn 50 step 不同点需确认 | ❓ |
| E. Prompt 格式导致模型输出乱码（与 LoRA 无关） | smoke 同样数据格式但成功，learn 用 hard 数据 | ❓ |

#### Miles 对比启发

| 方面 | Miles 做法 | Relax 当前做法 | 差异影响 |
|------|-----------|--------------|----------|
| 推送权重格式 | `bridge.export_adapter_weights()` → full 权重 | `all_gather` + `convert_to_hf` → full 权重 | **相同**：都推 FULL，SGLang 内部 TP shard |
| 模块命名 | `CanonicalLoRA` 可用分开 q/k/v 或 packed qkv | 直接推 `qkv_proj.lora_A/B` | **相同结果**：SGLang normalize 处理两种格式 |
| TP shard 逻辑 | SGLang `slice_lora_b_weights` 内部处理 | 同上 | **相同**：`QKVParallelLinearWithLoRA` 正确处理 GQA |
| weight sync 保护 | `torch_memory_saver.disable()` 包裹 | **同**：`torch_memory_saver.disable()` 包裹 (line 871) | 相同 |
| Grad buffer 保护 | `patch_param_grad_buffer_for_colocate_mode_lora()` | **无此补丁** | ⚠️ LoRA grad buffer 可能被 offload |
| Offload 处理 | `disable_param_buffers_cpu_backup=True` | `translate_gpu_to_cpu=True` (取 CPU backup) | 策略不同但目标相同 |

**Miles 独有关键补丁：`patch_param_grad_buffer_for_colocate_mode_lora()`**

```python
# miles/backends/megatron_utils/lora_utils.py:110-138
def patch_param_grad_buffer_for_colocate_mode_lora():
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.
    确保 LoRA adapter 的 DDP gradient buffer 不被 torch_memory_saver offload。
    """
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
    _original_init = _ParamAndGradBuffer.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs["disable_param_buffers_cpu_backup"] = True
        kwargs["disable_grad_buffers_cpu_backup"] = True
        _original_init(self, *args, **kwargs)
    _ParamAndGradBuffer.__init__ = _patched_init
```

此补丁的作用：在 colocate 模式下，`torch_memory_saver.pause(tag="default")` 会 offload 所有 "default" region 的 GPU 内存。LoRA 训练时基座冻结（`requires_grad=False`），DDP 只为 adapter 参数创建 gradient buffer。如果这些 buffer 也被 offload，后续恢复时可能导致状态不一致。Miles 通过此补丁将 adapter 的 param/grad buffer 标记为"不可 offload"。

#### 下一步

1. **最低成本验证**：在 smoke 配置（简单数据、2 step）的基础上只改为 `hard=True` 数据 → 确认是 LoRA/SGLang 问题还是数据/prompt 问题
2. **移植 Miles 的 grad buffer 补丁**：添加 `patch_param_grad_buffer_for_colocate_mode_lora()` 到 Relax → 排除 LoRA grad buffer offload 问题
3. **standalone LoRA 验证**（等 GPU quota 恢复）：在无 memory_saver 的独立 SGLang 中加载 zero-init LoRA → 确认是否为 SGLang LoRA 本身 bug

---

### 坑 23：根因定位 — `enable_weights_cpu_backup` 缺失导致基座模型丢失

**日期**：2026-06-01

#### 诊断测试结果

在 Modal 上运行的分层诊断测试（`diagnose()` 函数）证实：
- **Phase 1（简单数据 1+1=?）也产生完全乱码** — 模型输出均匀分布随机 token
- **从第一次 rollout 开始就是垃圾**（训练前！）
- LoRA 加载日志显示**成功**（4 个 TP rank 完成，target modules `[o_proj, qkv_proj]`）
- 192 个 adapter 张量正确同步

#### 根因分析

在 Relax 的 colocate 模式下：

| 步骤 | 操作 | 后果 |
|------|------|------|
| 1 | SGLang 加载基座模型 + LoRA | `enable_memory_saver=True`, `enable_weights_cpu_backup=False`（默认） |
| 2 | `release_memory_occupation` | `torch_memory_saver.pause(WEIGHTS)` — GPU 内存释放，**无 CPU 备份** |
| 3 | Megatron 训练 | 只更新 LoRA adapter 参数 |
| 4 | `onload_weights` → `resume_memory_occupation` | GPU 内存重新分配，但**内容为未初始化垃圾** |
| 5 | `load_lora_adapter_from_tensors` | 只推送 LoRA 权重（~192 tensors），基座模型仍为垃圾 |
| 6 | Rollout 生成 | **基座模型权重是垃圾 → 输出均匀分布乱码** |

**对比非 LoRA（全量权重同步）模式**：
- Step 4 后虽然也是垃圾，但 Step 5 会通过 `update_weights_from_ipc` 推送**全部权重**覆盖垃圾
- 所以非 LoRA 模式不需要 CPU backup

**对比 GenRM engine**：
- GenRM 引擎已正确设置 `"enable_weights_cpu_backup": True`（line 903）
- 主 rollout engine 遗漏了这个配置

#### 修复

```python
# Relax/relax/backends/sglang/sglang_engine.py
kwargs = {
    ...
    "enable_memory_saver": args.offload_rollout,
    # LoRA colocate: only adapter weights are synced after training, so
    # base model weights must survive the pause/resume cycle via CPU backup.
    "enable_weights_cpu_backup": (
        args.offload_rollout and getattr(args, "lora_enable", False)
    ),
    ...
}
```

#### 代价

启用 `enable_weights_cpu_backup` 会在 CPU RAM 中保留一份完整基座模型副本（~60GB for Qwen3-Omni bf16）。这是 LoRA colocate 模式的必要代价。

#### 端到端验证（已通过）

`modal_relax_smoke.py::verify_cpu_backup`（4×A100-80GB，~20 min）跑了完整 colocate LoRA RL 循环 2 个 step（train→offload→resume→热加载 LoRA→推理），用 `--dump-details` 落盘 rollout，再解码 token 做连贯性判定：

| Rollout | 时机 | 连贯样本 | 结论 |
|---------|------|---------|------|
| rollout 0 | 首次 offload→resume→热加载 LoRA→推理 | 32/32 | ✅ PASS |
| rollout 1 | 训练 step 0 后 再 offload→resume→热加载**更新后** LoRA→推理 | 32/32 | ✅ PASS |

证据：生成文本完全连贯且能答对（如 `Question: which letter is the 3-th option? ... <answer>C</answer>`），`rollout_log_probs ≈ -0.07 ~ -0.13`（正常区间，非乱码的均匀分布）。日志逐项确认 `release_memory_occupation`→`resume_memory_occupation` 200 OK、192 个 adapter 张量经 IPC 推送、4 个 TP rank 全部 `load_lora_adapter_from_tensors completes`。**坑 23 修复端到端验证通过：base 在 offload→resume→热加载后存活。**

（注：该次 toy 任务 `reward/advantages` 全为 0、无学习信号，仅验证「base 存活」；「LoRA 在学」由 `verify_lora_learning.py` 机制层单测单独验证，见文首「LoRA 在学：三层次状态」。）
