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

## 探针二结论：导出命名与 SGLang 的 parity（已验证，2026-08-11）

脚本 `mig_03_export_names.py` + `modal_probe_export_names.py`（T4，约 2 分钟；同样不建 Megatron 模型——命名由真实的 `mapping_registry` 推导，SGLang 侧喂按真实 config 算好形状的零张量）。

顺带确认了预构建镜像并不落后于 Relax `main` 的钉法：`sglang 0.5.12.post1`、`megatron.bridge 0.5.0`、`megatron.core 0.18.0`、`transformers 5.6.0`、`torch 2.11.0+cu129`。

导出侧的命名规则是 `linear_in → lora_A`、`linear_out → lora_B`，而 adapter 的 HF 名字由**基座权重的映射**推导（`_resolve_hf_adapter_param_name` → `mapping_registry.megatron_to_hf_lookup` → `_make_lora_param_name`），所以 Omni 的 `thinker.` 前缀是自动带上的。实测：

| Megatron 侧 | 导出的 HF 名字 |
|---|---|
| `...self_attention.linear_qkv.adapter.linear_in/out` | `thinker.model.layers.N.self_attn.{q,k,v}_proj.lora_{A,B}.weight` |
| `...self_attention.linear_proj.adapter.linear_in/out` | `thinker.model.layers.N.self_attn.o_proj.lora_{A,B}.weight` |

融合的 QKV `linear_out` 由上游 `_split_qkv_linear_out_weight` 调 `split_qkv_weights(model.config, ...)` 拆成 q/k/v，用的就是基座权重那套去交错逻辑。

SGLang 侧往返验证（30B-A3B thinker：hidden=2048、32 头、4 KV 组、head_dim=128，取 rank=32）：

- `get_layer_id` 是 `re.search(r"layers\.(\d+)\.")`，与前缀无关，带 `thinker.` 的名字照样解析出层号
- `normalize_qkv_proj` 把拆开的 q/k/v 堆回 `qkv_proj`：`lora_B` 得到 `(5120, 32)` = `32*128 + 2*4*128`，`lora_A` 得到 `(96, 2048)` = `3*32`，与内存池按 `max_lora_dim * 3` 开的 buffer 一致

**与 v1 的三点差异及影响：**

1. **拆开 vs 融合** —— v1 手写导出直接产 `qkv_proj.lora_{A,B}`；v2 产分开的 q/k/v，SGLang 的 `normalize_qkv_proj` 会堆回去，等价。
2. **QKV 去交错** —— v1 写了 `_reorder_qkv_lora_b` 手工把按 query group 交错的排布重排成 `[q;k;v]`；上游已经做了同样的事。**这段代码可以整个丢弃。**
3. **`base_model.model.` 前缀** —— v1 的名字带这个前缀，v2 不带（上游只在 `convert_adapter_weights_to_peft_state` 落盘时才加，而 Relax 的 `export_local_adapter` / `write_hf_peft_adapter` 都是原样用 `param_name`，没走那个函数）。对推给 SGLang 这条路无影响，因为 SGLang 全靠后缀和正则匹配；但对导出产物有影响，见下一节。

## 探针三结论：导出的 adapter 目录标准 PEFT 读不回来（已验证，2026-08-11）

脚本 `mig_04_peft_prefix.py` + `modal_probe_peft_prefix.py`（纯 CPU，约 2 分钟）。

起因是探针二发现 Relax 落盘的 key 不带 `base_model.model.` 前缀——`write_hf_peft_adapter` 是把 `export_adapter_weights` 的 `param_name` 原样 `save_file` 的，没走上游的 `convert_adapter_weights_to_peft_state`（那个函数才负责加前缀）。

先澄清一个**不成立的担心**：这不影响断点续训。`checkpoint.py` 的 `_save_lora_to_checkpoint` 写得很明确，`lora_adapter/` 是「可移植的导出产物，供外部/推理使用」，**不是续训来源**——LoRA 参数就是普通模型参数，原生 Megatron torch_dist checkpoint 已经存了。

但那句「例如用 `peft.PeftModel.from_pretrained` 加载」是不成立的。实测（peft 0.20.0 + transformers 5.6.0）：标准 PEFT 自己写出的 key 全部形如 `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`；把前缀去掉再读回，`PeftModel.from_pretrained` **不报错**，只发一条 `UserWarning: Found missing adapter keys`，然后 `lora_B` 全是零——也就是拿到一个「什么都没学到」的模型。

影响范围：

| 路径 | 是否受影响 | 原因 |
|---|---|---|
| 训练中推给 SGLang（内存 / 磁盘） | 否 | SGLang 按后缀和正则匹配，与前缀无关 |
| 断点续训 | 否 | 走原生 torch_dist checkpoint |
| 用标准 PEFT 加载导出的 adapter | **是** | 静默加载出全零 LoRA |

顺带印证了 v1 用的 `base_model.model.thinker.model.layers...` 命名本来就是符合 PEFT 规范的。

## sglang delta 移植（已完成，2026-08-11）

v1 的 delta 写在 0.5.9 上，共五处改动。逐条在 v0.5.12.post1（+ Relax vendor patch）上核对后，**四处仍然需要**：

| v1 的改动 | 在 v0.5.12.post1 上的状态 |
|---|---|
| `lora_manager` 的 `getattr` 兼容垫片 | 丢弃 —— 四个 ServerArgs 字段都已存在且默认值一致 |
| `lora_manager` 的 `should_apply_lora` 门 | 仍需要，且是上游回归（见下） |
| `tp_worker` 补 `monkey_patch_torch_reductions` | 仍需要 —— 上游只在 `update_weights_from_tensor` 里调了 |
| `patch_torch` 的 CPU tensor 保护 | 仍需要 —— 仍是无条件改第 7 个参数 |
| `qwen3_omni_moe` 的音频对齐 + LoRA 声明 | 仍需要 —— 上游该文件原封不动 |

**`should_apply_lora` 是上游回归**：v0.5.12.post1 里有 13 个模型定义了这个钩子（`qwen3_vl`、`qwen3_vl_moe`、`qwen2_vl`、`gpt_oss`、`gemma*` 等），`init_lora_modules` 的注释也还在说「embed_tokens 和 lm_head 要在 should_apply_lora gate 之前处理」，但**整个仓库没有任何调用方**——钩子成了死代码。Relax 的 vendor patch 没碰过 `lora_manager.py`，所以这是上游自身的状态。也就是说，v1 当年写的这段并不是 Omni 专属 hack，而是在补上游自己假设存在的行为。

移植后 `lora-omni-v2` 分支的构成（三个提交，叠在上游 tag `v0.5.12.post1` 上）：

1. `ce1717a786` —— Relax 的 vendor patch（44 文件），与官方 Docker 镜像保持一致
2. `b7225ce5ec` —— 三处上游修复：恢复 gate 调用、CPU reducer 保护、adapter 加载前装 torch reducer
3. `aa87211755` —— Omni 专属：外层类声明 LoRA 支持 + 变长音频对齐

拆成两个提交是为了将来能把第 2 个直接抽出来提给 sgl-project，不用再从 Omni 改动里剥离。

## 探针四结论：`_lora_pattern` 打在了哪些模块上（已验证，2026-08-11）

前三个探针都是训练侧（Megatron / Bridge）的，这个是推理侧的：把 sglang 的 `Qwen3OmniMoeForConditionalGeneration` 真建出来，看 `_lora_pattern` 对**真实模块名**的命中。

做法：只把配置压小（文本 48→2 层、专家 128→4、音频 32→2、视觉 27→2），在 meta device 上建模，不加载任何权重。T4 单卡两分钟。建模前要先 `initialize_dp_attention`，否则 `LayerCommunicator` 会在建 decoder layer 时报 `dp attention not initialized`。

121 个模块里，`--lora-target-modules qkv_proj o_proj` 的**纯后缀匹配命中 8 个，其中 4 个在塔里**：

```
thinker.visual.blocks.0.attn.qkv_proj              <-- 被门挡掉
thinker.visual.blocks.1.attn.qkv_proj              <-- 被门挡掉
thinker.audio_tower.layers.0.self_attn.qkv_proj    <-- 被门挡掉
thinker.audio_tower.layers.1.self_attn.qkv_proj    <-- 被门挡掉
thinker.model.layers.{0,1}.self_attn.{qkv_proj,o_proj}   <-- 放行
```

也就是说，没有 `should_apply_lora` 这个门，上游会把**一半**的目标模块挂到视觉塔和音频塔上——而 adapter 里根本没有这些模块的权重。这条正好是探针一（训练侧只在语言模型上挂 LoRA）的推理侧镜像，两边闭合。

顺带确认门本身放行的 8 个模块（`embed_tokens`、`lm_head`、每层的 `mlp.experts` 和两个投影）里，只有投影会真正落到 target 集合内，其余是 pattern 里的预留项，不会误伤。

脚本：`mig_05_sglang_lora_scope.py` + `modal_probe_sglang_scope.py`。注意 runner 会 clone fork 的 `lora-omni-v2` 分支并把 `python/` 顶到 `PYTHONPATH` 最前面，否则验的是镜像里预装的那份 sglang，看不到新加的门。

## 单元测试（已完成，2026-08-11）

v1 那份 `test_should_apply_lora_gate.py` 写在 `test/srt/lora/` 下，用的是 pytest 风格。0.5.12.post1 把测试重排到了 `test/registered/`（跑服务的）和 `test/registered/unit/`（不起服务的），并要求：镜像源码树的目录结构、文件顶部调 `register_cpu_ci` / `register_cuda_ci`、用 `unittest` + `CustomTestCase`、`__main__` 里不许裸调 `pytest.main`（仓库里有 `test_no_bare_pytest_main.py` 专门查这条）。所以按新约定重写，并拆成两份：

| 文件 | 内容 | 归属 |
|---|---|---|
| `test/registered/unit/lora/test_should_apply_lora_gate.py` | 通用 gate 行为：塔不被包、没有钩子的模型保持后缀匹配、全拒钩子什么都不包 | 随上游 PR 一起提，不含任何 Omni 依赖 |
| `test/registered/unit/models/test_qwen3_omni_lora_pattern.py` | `_lora_pattern` 的正负样例，模块名取自探针四的实测结果 | 我们的 delta |

拆开是为了让上游 PR 只带通用测试，不用捆 Omni 的改动。两份都靠 `LoRAManager.__new__` 跳过 `__init__`，不碰显存池也不下 adapter。跑法：`modal run modal_run_lora_tests.py`（挂本地测试文件到 fork 的 clone 上，改完不用先推）。
## 上游 PR（2026-08-11）

三个改动确认是上游的问题（而不是我们的适配），分别提了 PR。作者只署我自己。

| PR | 仓库 | 分支 | 状态 |
|---|---|---|---|
| Honor `should_apply_lora` when wrapping LoRA target modules | sgl-project/sglang | `fix/lora-honor-should-apply-lora` | 已提，#34428，等 CI 与 review |
| `fix(lora): expand path-pattern target modules to HF names` | redai-infra/Relax | `fix/lora-target-modules-wildcard-export` | 已推 fork，待开 PR |
| `fix(lora): write exported adapters in PEFT's key layout` | redai-infra/Relax | `fix/lora-adapter-peft-prefix` | 已推 fork，待开 PR |

正文分别在 `pr/sglang-01-body.md`、`pr/relax-01-wildcard-body.md`、`pr/relax-02-peft-prefix-body.md`。

Relax 那两个的动机各自都有硬证据：

- **通配符**：Relax 自己的 `scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-lora-mtp-8xgpu.sh` 就在用 `*decoder.layers.*.linear_qkv`，注释写明是为了让 MTP 层保持冻结。注入侧（Bridge）认这个模式，导出侧不认——glob 会原样落进 `adapter_config.json` 和 SGLang 启动参数，而两边都只按后缀匹配，等于导出的 config 一个模块都没点到。
- **PEFT 前缀**：`_save_lora_to_checkpoint` 的 docstring 和中英文档都承诺 `lora_adapter/` 可以用 `peft.PeftModel.from_pretrained` 加载，但落盘的 key 不带 `base_model.model.`。探针三实测：不报错，只警告一句 missing keys，然后 `lora_B` 全零。续训和 SGLang 都不受影响，受影响的恰好就是这个产物承诺的唯一用途。

CI 上踩过一个坑：Relax 的 `.pre-commit-config.yaml` 里有个本地 hook `docformatter --wrap-descriptions 79`，ruff 不管这条。#262 第一版就是因为测试文件里一段 docstring 按 ~90 列折行，被 docformatter 改写后判定「files were modified」而挂掉（Lint 和 ruff 全过，所以光跑 ruff 看不出来）。日志要登录才能下，用 `modal_precommit_relax_prs.py` 在容器里把仓库自带的 pre-commit 原样跑一遍就复现了。以后给 Relax 提 PR，docstring 描述段一律折到 79 列以内，或者直接跑那个脚本。
两个分支都做了双向验证（`modal_verify_relax_prs.py`）：打了补丁 47/48 个用例全过；把 `megatron_peft_utils.py` 换回上游 `main` 再跑，新加的用例全挂。`ruff format --check` 与 `ruff check` 干净（`modal_lint_relax_prs.py`，本地 pip 连不上源所以放容器里跑）。

## 待办

- [x] 探针一：LoRA 作用范围 —— 见上文
- [x] 探针二：`export_adapter_weights` 的命名与 v1 direct 导出 parity —— 见上文
- [x] 探针三：导出的 adapter 目录能否被标准 PEFT 读回 —— 不能，见上文
- [x] 把 v1 的 sglang delta 移植到 `v0.5.12.post1` —— 见上文
- [x] 探针四：`_lora_pattern` 对真实模块名的命中 —— 见上文
- [x] 给 sgl-project 开 PR：恢复 `should_apply_lora` 调用点 —— #34428
- [x] 补 gate 的单元测试 —— 已按上游新目录约定重写，6 个用例在 T4 上全过 —— 见上文
- [x] 给 Relax 开第一个 PR：`convert_megatron_to_hf_target_modules` 支持路径模式 —— 分支已推，正文已写
- [x] 给 Relax 开第二个 PR：`write_hf_peft_adapter` 补 `base_model.model.` 前缀 —— 分支已推，正文已写
- [ ] sglang 的另两处修复（`patch_torch` 的 CPU 张量越界保护、`tp_worker` 的 reducer 安装）单独提 PR
- [ ] 真机跑通 Omni Thinker LoRA adapter mode 的端到端 rollout