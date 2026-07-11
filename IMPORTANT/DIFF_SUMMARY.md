# Qwen3-Omni + LoRA 适配改动汇总（Diff Summary）

本文件汇总为把 **Qwen3-Omni（thinker）+ LoRA** 接入 RL 训练/推理，对上游 `Relax`、`sglang`
所做的改动，并判断**哪些是必须的、哪些是可选/任务相关的**。

> 这是人类可读的"改动索引 + 结论"。**精确、逐行、永不过期的 diff 以 git 为准**（见下方复现命令），
> 本文件不手抄逐行 diff，只做归类与解释。改动的"为什么/踩坑细节"另见 `LORA_RL_INTEGRATION.md`。

---

## 1. 上游基线与复现命令

| 组件 | 上游仓库 | 基线所基于的上游提交 | 本地 LoRA 快照提交 |
|---|---|---|---|
| Relax | `https://github.com/redai-infra/Relax.git` | `01973f3`（fix(megatron): restore cuda patch target in validate_args） | `86c97e0`（`lora-omni-baseline` 分支） |
| sglang | `https://github.com/sgl-project/sglang.git` | `19b60a4f9`（PR #26116，VLM reuse pretokenized ids） | `d8a1f27e7`（`lora-omni-baseline` 分支） |

复现精确 diff（任何时候都能重现，且和代码同步）：

```bash
# Relax：全部 LoRA 改动
git -C Relax diff 01973f3..86c97e0
git -C Relax show --stat 86c97e0        # 只看文件级 stat

# sglang：全部 LoRA 改动
git -C sglang diff 19b60a4f9..d8a1f27e7
git -C sglang show --stat d8a1f27e7
```

规模：Relax 17 文件 `+1001 / -41`；sglang 7 文件 `+554 / -16`。

分类图例：**【必须】** 核心适配，去掉就跑不通；**【修复】** 为跑通而必需的小 bug fix；
**【可选】** 任务相关或测试/脚本，与"omni+LoRA 适配"本身无关。

---

## 2. Relax 改动（17 文件）

### 2.1 LoRA 挂载 / 训练侧核心【必须】

| 文件 | 行数 | 作用 |
|---|---|---|
| `relax/backends/megatron/model_provider.py` | +138 | `apply_lora_to_model` / `wrap_model_provider_with_lora` / `_assert_lora_attached`；patch `LinearCrossEntropyModule` 注册。LoRA 挂载的核心。 |
| `relax/backends/megatron/model.py` | 16 | 按 `lora_enable` 在 LoRA wrapper 与原 freeze wrapper 间切换。 |
| `relax/backends/megatron/actor.py` | 55 | 集成 `UpdateLoRAFromTensor`，`lora_enable` 时走 LoRA 更新路径。 |
| `relax/utils/arguments.py` | +45 | 新增 `--lora-enable / --lora-rank / --lora-target-modules ...` 等 CLI。 |

### 2.2 权重同步（direct 路线核心）【必须】

| 文件 | 行数 | 作用 |
|---|---|---|
| `relax/backends/megatron/weight_update/update_lora_from_tensor.py` | +228 | **direct 路线的核心**：手写把 LoRA adapter 张量从 Megatron 推到 sglang（因为所钉 redai bridge 缺 `export_adapter_weights`，只能自己写）。 |
| `relax/backends/megatron/weight_conversion/qwen3_omni_moe.py` | +62 | `_convert_qwen3omni_lora_adapter` / `_reorder_qkv_lora_b`：把 Megatron 的 LoRA adapter 命名转成 sglang 命名 + qkv 重排（omni 专属）。 |
| `relax/backends/megatron/weight_update/hf_weight_iterator_direct.py` | 40 | 为 LoRA 参数加 `name_filter`；`all_gather` 前 `.contiguous()`。 |
| `relax/backends/megatron/weight_update/hf_weight_iterator_bridge.py` | 57 | `weight_type="lora"` 时走 `export_adapter_weights` 分支（bridge 路线，若 bridge 具备导出能力时用）。 |
| `relax/backends/sglang/sglang_engine.py` | +50 | `load_lora_adapter_from_tensors / load_lora_adapter / unload_lora_adapter`，以及 server args 加 `enable_weights_cpu_backup`（见踩坑 23：不加会丢基座）。 |

### 2.3 必需的小修复【修复】

| 文件 | 行数 | 作用 |
|---|---|---|
| `relax/backends/megatron/weight_update/common.py` | 14 | `all_gather` 前对参数分片 `.contiguous()`，否则同步出错。 |
| `relax/backends/megatron/__init__.py` | 4 | patch `patch_rotary_embedding`（omni 需要）。 |
| `relax/backends/megatron/arguments.py` | 2 | `rms_norm_eps` 校验小修正。 |
| `relax/engine/rollout/sglang_rollout.py` | 5 | rollout 请求带上 `lora_path`。 |

### 2.4 任务相关 / 脚本【可选】

| 文件 | 行数 | 作用 | 说明 |
|---|---|---|---|
| `relax/engine/rewards/bleu.py` | +115 | BLEU reward | 任务奖励，与 LoRA 适配无关 |
| `relax/engine/rewards/__init__.py` | 4 | 注册 BLEU reward | 同上 |
| `relax/engine/rewards/multiple_choice.py` | 4 | `extract_answer` 逻辑调整 | 任务相关 |
| `scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh` | +203 | LoRA 冒烟启动脚本 | 运行辅助，非核心代码 |

---

## 3. sglang 改动（7 文件）

### 3.1 omni + LoRA 推理核心【必须】

| 文件 | 行数 | 作用 |
|---|---|---|
| `python/sglang/srt/models/qwen3_omni_moe.py` | 50 | 声明 LoRA 支持；`packed_modules_mapping` + `_lora_pattern`；`should_apply_lora()`（**把多模态 tower 排除在 LoRA 之外，只挂 thinker**）；修复 audio feature padding。 |
| `python/sglang/srt/lora/lora_manager.py` | 28 | `server_args` 兼容取值（`getattr`）；接入 `should_apply_lora` 钩子。 |

### 3.2 TP 权重更新必需的修复【修复】

| 文件 | 行数 | 作用 |
|---|---|---|
| `python/sglang/srt/managers/tp_worker.py` | 1 | 反序列化前调用 `monkey_patch_torch_reductions()`。 |
| `python/sglang/srt/utils/patch_torch.py` | 7 | 给 `_REDUCE_TENSOR_ARG_DEVICE_INDEX` 加边界检查。 |

### 3.3 测试 / 辅助【可选】

| 文件 | 行数 | 作用 |
|---|---|---|
| `make_toy_lora.py` | +122 | 造玩具 LoRA 权重的小工具 |
| `test/srt/lora/_smoke_offline.py` | +123 | 离线冒烟脚本 |
| `test/srt/lora/test_should_apply_lora_gate.py` | +239 | `should_apply_lora` 门控单测 |

---

## 4. Relax（direct 路线）与 miles LoRA 的关键区别

> miles 的实现细节与逐行核实见 `LORA_RL_INTEGRATION.md` 的《Miles LoRA》章节，这里只列结论性差异。

| 维度 | miles | 本项目 Relax（direct 路线） |
|---|---|---|
| LoRA 挂载 | `lora_utils.py` + `bridge_lora_helpers.py`，走 **bridge PEFT** attach | `model_provider.py` 自己 `apply_lora_to_model` 挂载 |
| adapter 权重导出 | 依赖 bridge `export_adapter_weights` | **手写** `update_lora_from_tensor.py` 直接推张量（所钉 redai bridge 无该 API） |
| adapter 命名 → sglang | 由 bridge 导出统一处理 | 自己在 `qwen3_omni_moe.py` 做命名转换 + qkv 重排 |
| sglang 侧加载 | `load_lora_adapter` 系列 | 同样有 `load_lora_adapter` 系列（思路一致） |

**一句话**：miles 靠"bridge 官方导出"，本项目因所钉 bridge 版本缺该能力，用"手写张量推送"替代——
这正是本项目相对 miles **多出来的那部分工作**，也是当初考虑迁移上游 bridge 0.5.0 的动机
（迁移已评估后暂停，理由见 `MIGRATION_PLAN.md`）。

---

## 5. 结论：哪些必须 / 哪些多余

- **必须（omni+LoRA 适配的骨架，缺一不可）**：
  - Relax 侧 §2.1 + §2.2 + §2.3（挂载、direct 权重同步、omni 命名转换、若干 `.contiguous()`/patch 修复）。
  - sglang 侧 §3.1 + §3.2（omni 模型 LoRA 门控 + audio pad 修复 + TP 权重更新修复）。
- **可选 / 与适配无关（可按需保留或剥离）**：
  - Relax §2.4：BLEU reward、multiple_choice 调整、smoke 脚本 —— 属任务/运行辅助。
  - sglang §3.3：`make_toy_lora.py` 与两个测试 —— 验证辅助。
- **"多出来的"工作（版本妥协导致，非本质）**：
  - `update_lora_from_tensor.py` + `qwen3_omni_moe.py`（Relax 侧）的手写导出/命名转换。
    若迁到具备 `export_adapter_weights` 的上游 bridge，可由官方导出替代（见迁移方案，现暂停）。

---

_维护约定：改了 Relax/sglang 代码后，本文件的表格用 `git show --stat <新快照>` 重新核对；
逐行 diff 永远以 git 为准，不在此手抄。_
