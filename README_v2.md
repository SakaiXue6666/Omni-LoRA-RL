# v2 工作区 —— 迁到 Relax 官方 LoRA adapter 模式

**目标**：把 Qwen3-Omni Thinker 的 LoRA 强化学习，从 v1 自己实现的 direct 权重导出路线，迁到 Relax 官方已经支持的 LoRA adapter 模式。

**范围**：只做 Thinker LoRA + S2TT 文本输出。Talker / 语音链路整条冻结在 v1，不进 v2。

**验证平台**：Modal。

## 与 v1 的关系

v1 不是被替换，而是被冻结成 **parity oracle**：v2 每一步的行为都可以拿 v1 的已知结果对照。

| | v1 | v2 |
|---|---|---|
| 目录 | `d:\Li_Lab\RL\omni-lora-rl` | `d:\Li_Lab\RL\omni-lora-rl-v2`（本目录） |
| hub 分支 | `sglang_omni` @ `fd7915b`，tag `v1-frozen` | `v2`（基于 `main`） |
| submodule | Relax + sglang + sglang-omni | Relax + sglang |
| 镜像 | `slimerl/slime@sha256:bd219aba…`（= `nightly-dev-20260428a`） | `ghcr.io/redai-infra/relaxrl@sha256:8dc39af3…` |
| LoRA 权重通路 | 自己实现的 direct 导出 | Relax 官方 adapter 模式 |

v1 的全部细节见 `..\omni-lora-rl\README_v1.md`，代码 delta 另有三份存档 patch 在 `..\omni-lora-rl\patches\`。

本目录仍保留指向 v1 本地仓库的 `v1local` remote，随时可以 `git fetch v1local` 取回 v1 的任何提交。

## 为什么基于 main 分支

GitHub 仓库原本就是这么分的：`main` = Relax + sglang，`sglang_omni` = Relax + sglang + sglang-omni。v2 的范围正好等于 `main` 的结构，所以直接从 `main` 起，而不是从 `sglang_omni` 上剥掉一个 submodule。

从 `sglang_omni` 额外带过来两份训练侧记录作参考：`IMPORTANT/OMNI_RELAX_HANDOFF_2026-07-18.md` 和 `IMPORTANT/OMNI_STREAMING_GPU_2026-07-19.md`。语音脚本、v1 冻结产物都留在 v1，不带。

## submodule 基线

两个 submodule 都钉在**上游新版**上，这样我们的改动将来可以直接对上游开 PR。

| submodule | 上游 | v2 基线 | v2 分支 |
|---|---|---|---|
| Relax | `redai-infra/Relax` | `main` @ `9a5674af` | `lora-omni-v2` |
| sglang | `sgl-project/sglang` | tag `v0.5.12.post1`（`5a15cde858`） | `lora-omni-v2` |

sglang 的版本不能随便选新的：Relax 的 Dockerfile 里 `BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1-cu129`，而且 Relax **自己也给 sglang 打补丁**——`docker/patch/latest/sglang.patch` 是软链，指向 `docker/patch/sglang/v0.5.12.post1.patch`（135 KB，改 44 个文件），构建时 apply 到镜像里的 `/sgl-workspace/sglang`。选别的版本这份补丁就打不上。

### sglang 的分层约定

| 层 | 内容 | 提交 |
|---|---|---|
| 0 | 上游 tag `v0.5.12.post1` | `5a15cde858` |
| 1 | Relax 的 `v0.5.12.post1.patch`（44 文件，原样 vendor） | `ce1717a786` |
| 2+ | 我们的 Qwen3-Omni LoRA delta | 待移植 |

这样整个 submodule 可以直接挂载进容器而不丢 Relax 的改动；将来开 PR 时只取第 2 层起的 diff，对上游仍然是干净的。

v1 的 delta 一共 7 个文件，和 Relax 那 44 个文件只有 `python/sglang/srt/managers/tp_worker.py` 一处重叠，核心的 `models/qwen3_omni_moe.py`、`lora/lora_manager.py`、`utils/patch_torch.py` 上游补丁完全没碰，分层很干净。

本地还保留着 v1 的 sglang 状态：分支 `lora-omni-baseline`、tag `v1-frozen`（`d13903a9`），移植时可以直接对照。

## 环境验证（已完成，2026-08-10）

在 `ghcr.io/redai-infra/relaxrl@sha256:8dc39af3…` 上跑通了 Phase 0.1 的四条判据，以及 0.1b（注册 Qwen3-Omni bridge 后复验识别）：

- `megatron.core` 与 `megatron.bridge` 共存
- `AutoBridge` 具备 adapter 导出 API
- PEFT LoRA 可导入
- `AutoBridge` 能识别 Qwen3-Omni

这正是 2026-07 那次迁移卡住的地方——当时 Megatron-Bridge 0.5.0 和 megatron-core 装不到一起。新版 Relax 的 Dockerfile 用 `3rdparty/Megatron-LM` submodule + rsync 的方式解决了，所以这条路现在通了。

相关脚本：`mig_00_env.py`（判据 1-4）、`mig_01_omni_bridge.py`（注册后复验）、`modal_env_gate_v2.py`、`modal_gate_omni.py`。

探针用的是「预构建镜像 + 指定源码」：镜像照用官方的，Relax 源码走 `PYTHONPATH` 注入，省掉每次改动都要重建镜像。

## v1 → v2 的关键差异

| v1 怎么做 | v2 对应的 Relax 官方设施 |
|---|---|
| 自己写 direct 权重导出 | `relax/backends/megatron/weight_update/lora_adapter_sync.py` |
| 自己实现 LoRA 注入 | `relax/utils/megatron_peft_utils.py` 的 `apply_lora_to_model` |
| 自己做 adapter 转换 | Megatron-Bridge 的 `export_adapter_weights` |
| —— | 参考脚本 `scripts/training/text/run-qwen3-4B-lora-adapter-x8gpu-async.sh` |
| —— | 参考测试 `tests/backends/megatron/weight_update/test_lora_weight_sync.py` |

sglang 侧不一样：上游至今没有 Omni 的 LoRA 支持，`qwen3_omni_moe.py` 里既没有 `should_apply_lora` 也没有 audio/vision tower 的 LoRA 排除逻辑。这部分是**永久 delta**，不是技术债，需要一直带着（也正是将来给 sgl-project 开 PR 的内容）。

## 探针一结论：LoRA 的作用范围（已验证，2026-08-11）

脚本 `mig_02_lora_scope.py` + `modal_probe_lora_scope.py`，在 v2 镜像 + Relax `9a5674af` 上跑通（T4，约 2 分钟；塔用 meta device 构建，不加载权重）。

Relax 的 Megatron 模型 `Qwen3OmniMoeModel` 在 `pre_process` 的 rank 上确实建了两个塔，而且用的是 **transformers 的 HF 实现**（`Qwen3OmniMoeAudioEncoder` / `Qwen3OmniMoeVisionEncoder`），只有 `language_model` 是 Megatron 的 GPT。`PEFT.__call__` 走的是通用的 `_walk_model`，会下探到 HF 子模块，所以塔在遍历范围内——挂不挂上完全取决于名字撞不撞。

实测（transformers 5.6.0）：

| 塔 | 线性层叶子名 | 与 Megatron 命名是否撞车 |
|---|---|---|
| audio | `q_proj` `k_proj` `v_proj` `out_proj` `fc1` `fc2` `proj1` `proj2` `conv_out` | 全不撞 |
| vision attention / merger | `qkv` `proj` `0` `2` | 不撞 |
| vision MLP | `linear_fc1` `linear_fc2` | **撞** |

命中结果：

- `--lora-target-modules linear_qkv linear_proj`（v1 与官方默认，即 Q/K/V/O）→ 两个塔**命中 0 个模块**，LoRA 只落在 language model 上。
- 追加 `linear_fc1 linear_fc2` → 命中 `vision_model.blocks.N.mlp.linear_fc1/2`，即视觉塔每一层的 MLP（探针把 depth 压到 1，实际 27 层就是 54 个模块）。

**结论：只挂注意力投影时不需要改 Relax 的注入逻辑，CLI 默认值就是对的。** 但这是个隐式约束——哪天想给 MLP 加 LoRA，必须先用通配符（`ModuleMatcher` 支持 `*.layers.0.*.linear_qkv` 这种）或 `exclude_modules` 把视觉塔排除掉，否则会静默给视觉塔挂上 adapter，而 SGLang 基座那边根本没有对应模块。

## 待办

- [x] 探针一：LoRA 作用范围 —— 见上节
- [ ] 探针二：`export_adapter_weights` 的张量命名与 v1 direct 导出做 parity 对照
- [ ] 把 v1 的 sglang delta（7 文件，写在 0.5.9 上）移植到 `v0.5.12.post1`
- [ ] 给 Relax 开第一个 PR：让 `convert_megatron_to_hf_target_modules` 支持通配符（现在通配符会原样落进 `adapter_config.json`，SGLang 的 PEFT 加载器不认 glob）