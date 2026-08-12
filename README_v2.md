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
## 上游 PR（2026-08-11 起）

五个改动确认是上游的问题（而不是我们的适配），分别提了 PR。作者只署我自己。

| PR | 仓库 | 分支 | 状态 |
|---|---|---|---|
| Honor `should_apply_lora` when wrapping LoRA target modules | sgl-project/sglang | `fix/lora-honor-should-apply-lora` | 已提，#34428，等 CI 与 review |
| `fix(lora): expand path-pattern target modules to HF names` | redai-infra/Relax | `fix/lora-target-modules-wildcard-export` | 已提 #261，CI 全绿 |
| `fix(lora): write exported adapters in PEFT's key layout` | redai-infra/Relax | `fix/lora-adapter-peft-prefix` | 已提 #262，修掉 docformatter 后 CI 全绿 |
| `fix(lora): inline adapter tensors into the engine payload` | redai-infra/Relax | `fix/lora-adapter-transport-shm` | 分支已推 `1ceb779b`，正文已写 |
| Fix IndexError when reducing CPU tensors after `monkey_patch_torch_reductions` | sgl-project/sglang | `fix/reduce-tensor-cpu-guard` | 分支已推 `11093f14`，正文已写 |

正文分别在 `pr/sglang-01-body.md`、`pr/relax-01-wildcard-body.md`、`pr/relax-02-peft-prefix-body.md`、`pr/relax-03-transport-body.md`。

Relax 那两个的动机各自都有硬证据：

- **通配符**：Relax 自己的 `scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-lora-mtp-8xgpu.sh` 就在用 `*decoder.layers.*.linear_qkv`，注释写明是为了让 MTP 层保持冻结。注入侧（Bridge）认这个模式，导出侧不认——glob 会原样落进 `adapter_config.json` 和 SGLang 启动参数，而两边都只按后缀匹配，等于导出的 config 一个模块都没点到。
- **PEFT 前缀**：`_save_lora_to_checkpoint` 的 docstring 和中英文档都承诺 `lora_adapter/` 可以用 `peft.PeftModel.from_pretrained` 加载，但落盘的 key 不带 `base_model.model.`。探针三实测：不报错，只警告一句 missing keys，然后 `lora_B` 全零。续训和 SGLang 都不受影响，受影响的恰好就是这个产物承诺的唯一用途。

- **adapter 传输**：真机 4×A100 上第一次推 adapter 就死在 TP0，而 TP1–3 成功。根因是
  payload 里装的是 `/dev/shm` 引用不是字节，见下文探针九。这个 PR 把序列化抽成
  `megatron_peft_utils.serialize_adapter_tensors` 再改内联——抽函数不是为了好看，是因为
  原调用处所在的 `update_weight_from_tensor.py` 模块级就 `from megatron.core import mpu`，
  CPU CI 里 import 不进来，测试只能 skip，等于没有保护。挪到纯 torch 的工具模块后，
  测试落在已经跑在 CI 里的 `tests/utils/test_megatron_peft_utils.py`。

- **CPU 张量越界**：`monkey_patch_torch_reductions()` 换掉的是**所有**张量的 reducer，不只
  CUDA 的，而 `_reduce_tensor_modified` 无条件改写第 6 个参数——CPU 张量归约出来的元组根本
  没有那一位，于是 `IndexError`。这条不是我们独有：verl 的
  [#4065](https://github.com/volcengine/verl/issues/4065) 从 2025 年 11 月挂到现在，多人复现，
  traceback 一模一样，帖子里流传的临时补丁就是这个长度守卫。也就是说下游用户现在要么手改
  site-packages，要么改走 merge 模式绕开推 adapter。

CI 上踩过一个坑：Relax 的 `.pre-commit-config.yaml` 里有个本地 hook `docformatter --wrap-descriptions 79`，ruff 不管这条。#262 第一版就是因为测试文件里一段 docstring 按 ~90 列折行，被 docformatter 改写后判定「files were modified」而挂掉（Lint 和 ruff 全过，所以光跑 ruff 看不出来）。日志要登录才能下，用 `modal_precommit_relax_prs.py` 在容器里把仓库自带的 pre-commit 原样跑一遍就复现了。以后给 Relax 提 PR，docstring 描述段一律折到 79 列以内，或者直接跑那个脚本。
两个分支都做了双向验证（`modal_verify_relax_prs.py`）：打了补丁 47/48 个用例全过；把 `megatron_peft_utils.py` 换回上游 `main` 再跑，新加的用例全挂。`ruff format --check` 与 `ruff check` 干净（`modal_lint_relax_prs.py`，本地 pip 连不上源所以放容器里跑）。

## 探针六：CPU 张量的 reduce 越界（2026-08-11，纯 CPU）

给第三个 sglang PR 攒的实证，顺便解释了为什么 `patch_torch` 和 `tp_worker` 两处改动必须捆在一起提。

上游 `_reduce_tensor_modified` 无条件改写参数元组的 index 6，注释里的依据是「签名多年没变」。那个假设只对 CUDA 张量成立。实测一个 CPU 张量经 `reductions.reduce_tensor` 出来是：

```
rebuild 函数: rebuild_tensor      参数元组长度: 3
  [0] _TensorMeta   [1] TypedStorage   [2] (0, torch.Size([4, 4]), (4, 1), False)
```

长度 3，取 index 6 直接 `IndexError: tuple index out of range` —— 不是静默写坏某个字段，是硬崩。A/B 三段都成立：守卫版序列化通过、换成上游那版当场抛 IndexError、换回守卫版又能通过。

因果链值得写清楚，否则容易误以为这是上游的既有 bug：v1 在 0.5.9 上推 CPU 张量没事，是因为当时 LoRA 那条路压根没调 `monkey_patch_torch_reductions`；是我们给 `tp_worker` 补上 reducer 安装之后，越界才暴露出来。所以这两处是同一个改动的两半，要一起提。verl 踩过同一个坑（bug #4065）。

脚本：`mig_07_cpu_reduce.py`，入口 `modal run modal_probe_serve_lora.py --stage reduce`（CPU，约 90 秒）。

## 探针五：真机上把 adapter 热推给 sglang（2026-08-11，1×A100-80GB）

第一次在真硬件上跑通 v2 的推理侧。判据沿用 v1 的三条 —— 只判「输出变了」太弱，改一个字节的权重输出也会变。

| 检查 | 结果 |
|---|---|
| 带 `enable_lora` 起 Omni | 起来了。这一步本身就是 gate 的验证：塔要是被误包，加载时 hidden dim 就对不上 |
| base 生成 | `今天天气很好。`，套了 chat 模板（v1 教训：不套会直接吐 `<|im_end|>`） |
| 可逆性：推 B=0 的 adapter | 与 base 的 logprob **最大差 0.000000** |
| 生效性：换非零 B（scale 0.12） | 最大差 2.416，输出明显跑偏 |
| 稳定性：连续 unload 到 load 三轮 | 不崩，空闲显存稳定在 11.5/79.3 GiB |

可逆那条是这次最有价值的一条：B=0 时精确复现 base，说明 384 个张量的命名全部被 sglang 认了下来。要是有名字没对上，那部分权重会被静默丢弃，B=0 时同样看不出差别 —— 但反过来，能精确复现、同时非零 B 又确实生效，两条合起来才排除了「名字错了所以没生效」和「名字错了但恰好没影响」这两种假通过。这把探针二的静态命名比对升级成了真机验证。

adapter 全程是 CPU 张量（384 个），走的正是探针六那条序列化路径。

脚本：`mig_06_serve_lora.py`，三个 stage：`--stage inspect`（CPU，查模型卷与源码）、`--stage reduce`（CPU，探针六）、`--stage probe`（A100，本节）。设计直接沿用 v1 `modal_run.py` 的实验 G/J：引擎参数 `disable_cuda_graph=True, mem_fraction_static=0.85, tp_size=1`、chat 模板、adapter 构造方式、logprob 比对，都是 v1 已经验过的，这次没有重新试错。

## 探针七：训练侧冒烟（2026-08-12，1×A10G，约 3 分钟）

前六个探针验的全是推理侧和导出命名的静态预测，训练侧在 v2 上一次都没跑过。这一步只回答三件事，完全不碰 rollout、数据集、优化器。

省钱的关键是缩层：`provider.num_layers = 2`，随机初始化，不加载 30B 真实权重。这不是我们发明的 hack —— Relax 的 `model_provider.py` 把 `num_layers` 列进了 `bridge_keys`，注释写明「Allow CLI to override layer count for layer-reduced training」，是官方支持的路子。缩完 3.06B，一张 A10G（22 GiB）绰绰有余，用不着 A100。

| 检查 | 结果 |
|---|---|
| `AutoBridge` 建 Omni | `Qwen3OmniModelProvider` → `Qwen3OmniMoeModel`，48 层缩到 2 层 |
| LoRA 注入范围 | 8 个 adapter 参数，每层 4 个（`linear_qkv` / `linear_proj` 各 A/B），全在 `language_model.*`，塔里一个都没有 |
| 底模冻结 | 可训参数恰好就是那 8 个 adapter |
| adapter 导出 | 16 个张量，命名与形状见下 |

导出的命名和形状：

```
thinker.model.layers.{0,1}.self_attn.q_proj.lora_A.weight  (32, 2048)
thinker.model.layers.{0,1}.self_attn.q_proj.lora_B.weight  (4096, 32)
thinker.model.layers.{0,1}.self_attn.k_proj.lora_B.weight  (512, 32)
thinker.model.layers.{0,1}.self_attn.o_proj.lora_A.weight  (32, 4096)
```

三条结论：

1. **Megatron 侧是 fused 的 `linear_qkv`，导出时被 Bridge 的 `QKVMapping` 拆成了 `q/k/v_proj`。** 这正是 v1 手写 de-interleave 干的事，现在上游做了 —— 探针二是静态推断，这次是真模型上的实证，那段代码可以退休。
2. **形状与 v1 写死的值逐个对上**：hidden 2048、q_out 4096、kv_out 512。
3. **数量闭环**：2 层导出 16 个，48 层就是 384 个 —— 正好是探针五推给 sglang 引擎的那 384 个张量。训练侧产出什么、推理侧吃什么，两边咬合上了。

`convert_megatron_to_hf_target_modules(['linear_qkv', 'linear_proj'])` 落到 `adapter_config.json` 里是 `['q_proj', 'k_proj', 'v_proj', 'o_proj']`，与导出的叶子模块完全覆盖。

脚本：`mig_08_train_side.py` + `modal_probe_train_side.py`。用的是 Relax 自己的 `build_lora_peft` 和 bridge，不是重写一遍，所以验的是真实代码路径。

## 与 v1 的对照（2026-08-12，做端到端训练之前的核对）

把 v1 训练侧翻了一遍，和探针七的结果逐条对，有五处需要记下来。

**1. LoRA 超参不一致，端到端时要改回 v1 的。** v1 冒烟脚本用的是 `--lora-rank 16 --lora-alpha 32`（sglang 侧 `--sglang-max-lora-rank 16`），探针五和七我用的是 32/64。功能上都成立，但 v1 那条 100 步 S2TT 的 BLEU 曲线是在 16/32 上跑出来的，v2 想和它对比就得对齐超参，否则说不清差异来自迁移还是来自 rank。

**2. `--lora-target-modules` 的默认值是 v1 的一处 delta，我们没有移植，而且不该移植。** v1 把默认改成了 `*language_model*linear_qkv` / `*language_model*linear_proj`，理由是裸的 `linear_qkv` 会挂到 `audio_model`。但 v1 那个结论出自 `verify_lora_attach.py` 里手搭的 Omni-like 树，不是真模型。探针七在 Relax 真建出来的 `Qwen3OmniMoeModel` 上用裸名字，8 个 adapter 参数全在语言模型里，塔干净 —— 因为 v2 的 audio/vision 是 HF 模块，叫 `q_proj/k_proj/v_proj`，压根不叫 `linear_qkv`。

   而且这里有个反向依赖：**通配符恰好是我们给 Relax 提 [#261](https://github.com/redai-infra/Relax/pull/261) 要修的那个 bug**（glob 会原样落进 `adapter_config.json`，导出侧一个模块都点不到）。所以真要用 v1 的加固写法，就得先带上 #261；用裸名字则不需要。结论是保持裸名字，不把 #261 变成端到端训练的前置条件。

**3. v1 那个"adapter 名里带 audio/vision 就告警"的守卫（`_assert_lora_attached`）在 v2 上游不存在。** 探针七证明当前不需要它，但它便宜，可以作为后续给 Relax 的一个小 PR，或者先在我们这侧留个断言。

**4. v1 根本没有 `export_adapter_weights`。** 它 pin 的 redai bridge（`f13bec09`）早于这个 API，导出走的是 `convert_qwen3omni_to_hf` 的 direct 路线，里面还得手写 `_reorder_qkv_lora_b()` 做 GQA 维度重排。探针七证明 v2 的 bridge 原生就把 fused `linear_qkv` 拆成 `q/k/v_proj` —— 这一整条 direct 路径连同重排函数，在 v2 里整个删掉。这是这次迁移最大的一块减法。

**5. 下一个风险点是 TP>1 的 adapter gather，探针七只验了 TP=1。** v1 在这儿流过血（坑 16：TP=4 手写 all_gather 撞 CUDA illegal access），并且专门留了 `verify_lora_tp.py`（TP=2）和 `verify_tp_gather.py`（TP=4）两个探针。v2 这块交给 bridge 做，大概率没事，但"大概率"不算验过 —— 上 4×A100 之前值得先花两张卡验一次导出 parity。

端到端时要照抄的 v1 配置：`TP=4 / EP=4 / PP=1`，**关掉 sequence-parallel 和 recompute**（v1 记录：recompute + SP + LoRA 会让 `lora_B` 的 backward 出 NaN），`A100-80GB:4`、timeout 240 分钟、`retries=10`（Modal 抢占后从卷上的 ckpt 续跑）。

## 探针八：TP=2 的 adapter 导出 parity（2026-08-12，2×A10G，约 5 分钟）

探针七只验了 TP=1，而 v1 恰恰在 TP>1 的 adapter gather 上流过血（坑 16：TP=4 手写 all_gather 撞 CUDA illegal access），当年为此留了两个专门的探针。v2 把 gather 交给 bridge，理应没事——但"理应"不算验过。

判据设计上有个坑必须绕开：**不能拿 TP=1 和 TP=2 的导出直接比数值**。Megatron 的 TP 初始化按 rank 分 RNG 种子，同一个逻辑权重在两种并行度下本来就不是同一份随机数，比出来的差异毫无意义。改成两条自洽判据：形状必须是完整尺寸；导出的 q/k/v 三块拼起来，必须和我们手工 `all_gather` 出来的 fused 矩阵是同一批行。

结果全过，顺带看清了分片长什么样——这是这次最有价值的发现：

```
linear_qkv.adapter.linear_in   (16, 2048)   partition_dim=0   <-- rank 维被切了！32 -> 16
linear_qkv.adapter.linear_out  (2560, 32)   partition_dim=0        输出维切，5120 -> 2560
linear_proj.adapter.linear_in  (32, 2048)   partition_dim=1        输入维切
linear_proj.adapter.linear_out (1024, 32)   partition_dim=0
```

**LoRA 的 rank 维本身也参与 TP 切分**（`linear_in` 每卡只有 16 行，两卡合起来才是 rank=32），这点之前没意识到。导出的 `q_proj.lora_A` 是完整的 `(32, 2048)`，说明 bridge 把这一维也正确拼回来了。三种不同的切分模式（rank 维、输出维、输入维）在一次导出里全部还原正确。

对账细节：手工 gather 的 fused `linear_out` 是 `(5120, 32)`，导出的 q+k+v 拼接也是 `(5120, 32)`，两者排序后逐元素相等；且 `q_proj` 的每一行都能在 fused 矩阵里找到——gather 没丢数据，GQA 的 de-interleave 也没错位。

一个观察：TP>1 时 `Qwen3OmniModelProvider.finalize()` 会强制打开 `sequence_parallel`。这次没跑 backward 所以无所谓，但 v1 记录过 recompute + SP + LoRA 会让 `lora_B` 的 backward 出 NaN，端到端时要注意这个交互。

脚本：`mig_09_tp_export.py` + `modal_probe_tp_export.py`（torchrun 起 2 进程）。

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
- [x] 探针六：CPU 张量 reduce 越界的 A/B —— 上游 IndexError，守卫版通过，见上文
- [x] 探针五：1×A100 上验 adapter 热加载、可逆、稳定 —— 五项全过，见上文
- [x] sglang 的 `patch_torch` CPU 张量越界保护 —— 分支已推，正文已写；`tp_worker` 那处在 main 上已作废（上游把反序列化收敛进 `_deserialize_own_rank`，里面就装了 reducer，LoRA 那条路也走它）
- [x] 探针七：训练侧冒烟（Bridge 建 Omni + LoRA 注入 + adapter 导出）—— 一次通过，见上文
- [x] 探针八：TP=2 的 adapter 导出 parity —— 三种切分模式全部正确还原，见上文
- [x] 探针九：adapter 传输三路对照 —— 复现真机 ENOENT，内联字节全过，见下文
- [x] 给 Relax 开第三个 PR：adapter 推送改内联字节 —— 分支已推，正文已写
- [x] 真机跑通端到端训练 —— 4×A100 跑完 40 步 S2TT，逐段贴着 v1 的曲线，见下文
- [ ] 同传 S2TT 的移植 —— v1 的 `examples/simul_s2tt/rollout.py` 走 `--custom-generate-function-path`，务必连 `3a6eb2f` 那个 reward 去污染补丁一起带过来
- [ ] 续训不生效的排查 —— 见下文「两条运维教训」

## 端到端训练：从 v1 移植过来的那套

目标是回答一件事：迁到 v2 之后还学不学得动。所以超参逐项照抄 v1 那次 100 步 S2TT
实验，reward 也逐字移植——口径动一点，曲线就没法比。v1 的参照曲线：前 10 步 BLEU
均值约 0.29，31-40 步约 0.41，80-90 步约 0.53。判据必须用窗口均值，因为 v1 自己的
单步波动能有 ±0.1（step 90 是 0.539，step 100 掉到 0.397）。

首跑定 40 步：v1 记录每步约 4.4 分钟，40 步约 3 小时，卡在一个 240 分钟的容器窗口
里跑得完；而 30-40 步正好是信号浮出噪声的位置。100 步要靠 SAVE_INTERVAL + retries
跨容器续跑才撑得下来，验证迁移不需要付这个成本。

三个新文件：

- `omni_s2tt/bleu_rm.py` —— 句级 BLEU，挂在 `--custom-rm-path` 上。v1 是直接改
  Relax 源码加了个 `rm_type=bleu` 分支，v2 有扩展点，这条 delta 从"改框架"降级成
  "加一个文件"。算法与 v1 逐字一致，故意不做改进。只加了一处观测：统计响应里带
  `<|...|>` 的条数（v1 在同传里发现过这种污染会把 BLEU 压到真实值的约四成），但不
  改分数，因为 v1 的 S2TT 基线当时也没打这个补丁。
- `omni_s2tt/run-qwen3-omni-lora-s2tt-4gpu.sh` —— v1 冒烟脚本的 v2 版。
- `modal_train_s2tt.py` —— runner，含 CPU 版 `check`、断点续跑、BLEU 曲线判定。

### 相对 v1 少掉的三样东西（都是上游变好了）

1. **LoRA 开关**：v1 的 `--lora-enable --lora-name policy` 在 v2 不存在。LoRA 由
   `--lora-rank > 0` 打开，adapter 名字是代码常量 `LORA_ADAPTER_NAME`，再加
   `--lora-adapter-mode` 走 adapter 模式。
2. **sglang 的 LoRA 参数**：v1 要手写 `--sglang-enable-lora` /
   `--sglang-max-lora-rank` / `--sglang-lora-target-modules`，v2 的
   `sglang_engine.py` 自己从训练侧参数推（target 经
   `convert_megatron_to_hf_target_modules` 转成 `q/k/v/o_proj`），rollout 也自动带
   `lora_path`。整条链路是上游接好的。
3. **一堆绕行**：`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`、`--sglang-attention-backend
   triton`、`--sglang-disable-cuda-graph`、`--sglang-disable-custom-all-reduce`。这些
   在 v1 是因为往旧 slime 镜像里注入了更新的 sglang，内核二进制对不上；v2 用 Relax
   官方镜像 + 同版本(0.5.12.post1)的 sglang fork，不存在这个错配。

保留 v1 的一条安全默认：**不开 sequence-parallel、不开 recompute**（v1 记录过
recompute + SP + LoRA 会让 lora_B 的 backward 出 NaN）。注意 TP>1 时 Omni 的
provider 会在 `finalize()` 里自己打开 sequence_parallel，这里不额外叠加。

`--lora-target-modules` 用裸名字 `linear_qkv linear_proj` 而不是 v1 的通配符：探针七
确认过塔不会被误挂（v2 的 audio/vision 是 HF 模块，叫 `q_proj` 之类，压根不叫
`linear_qkv`），而通配符会原样落进 `adapter_config.json`（已提 Relax #261）。

### 数据

复用 v1 的 `s2tt-data` 卷，128 条 FLEURS en→zh，与 v1 当年同一批，音频零缺失。
样本形如 `{"prompt": "<audio>...", "audios": [wav], "label": {"ground_truth": ...},
"metadata": {"tgt_lang": "zh", ...}}`。128 条配 rollout-batch 8 是每 16 步一个 epoch，
40 步约 2.5 个 epoch —— v1 跑 100 步时是同样的数据量，所以可比。

## 探针九：adapter 传输的三路对照（2026-08-12，纯 CPU，几分钟）

第一次起 40 步训练时，模型建好、LoRA 注入、引擎起来、基础权重同步完成（19743 个参数、
7.7 秒），死在第一次推 adapter。而且只死一个 rank：TP1–TP3 都打了
`loading from tensors completes`，只有 TP0 报

```
RuntimeError: unable to open shared memory object </torch_4103_2908601601_198>
in read-write mode: No such file or directory
```

`4103` 是训练侧 rank 0 的 pid。**这暴露了探针五的盲区**：探针五验的是进程内
`sglang.Engine` 的热加载，根本没经过跨进程共享内存这条路，所以「adapter 能热推」这个
结论在真实的 Ray → HTTP 链路下并不自动成立。

用一个几乎免费的 CPU 探针（`modal_probe_transport.py`：一个 Ray actor 序列化，四个
consumer actor 反序列化）把三种传输方式摆在一起：

| 传输方式 | 结果 |
|---|---|
| `file_descriptor`（torch 默认） | 四个 rank 全挂，`AuthenticationError` |
| `file_system`（上游实际在用） | TP0 挂在 ENOENT，TP1–3 成功 |
| pickle + base64 内联（v1 的做法） | 四个 rank 全过，校验和都对得上 |

第二行和真机报错一字不差。**关键是必须让 TP0 迟到才能复现**：`/dev/shm` 那个文件是
引用计数管理的，先到的 rank 映射完、返回时释放引用，计数归零文件即被 unlink，落在后面
的 rank 再去开就 ENOENT。第一次跑探针时四个消费者几乎同时进来，反而全过；加 8 秒延迟
立刻复现。这解释了真机上为什么偏偏只有 TP0 死，也说明这个 bug 一直是靠调度运气活着的。

值得记一笔：上游的注释显示他们**已经踩过一次**——默认策略跨不过 Ray→HTTP，所以主动切成
了 `file_system`。真机死的是绕完之后的第二个坑。两种策略的共性才是问题所在：payload 里
放的是引用，而引用的有效期取决于生产者还攥不攥着那块存储。

修法是 adapter 那一路改成内联真字节，payload 从 0.1 MB 变成 31.6 MB（rank 16 的 adapter
约 24 MB，v1 扛着这个代价跑完了 100 步）。sglang 一侧不用动，因为
`MultiprocessingSerializer.deserialize` 本来就是 base64 解码 + unpickle。基础权重那一路
不碰：那些张量在设备上，序列化成 CUDA IPC 句柄，本身自包含——这也正是基础权重同步从来
没出过事的原因。

## 端到端 40 步 S2TT（2026-08-12，4×A100-80GB，colocate TP4/EP4）

跑完 40/40 步，每步推一次 adapter，没有再出传输错误。按十步分段与 v1 对照
（v1 表是 1 起步、我们是 0 起步，已对齐）：

| 区间 | v1 | v2 | 差 |
|---|---|---|---|
| 前 10 步 | 0.287 | 0.294 | +0.007 |
| 第 11–20 步 | 0.344 | 0.331 | −0.013 |
| 第 21–30 步 | 0.345 | 0.357 | +0.012 |
| 第 31–40 步 | 0.389 | 0.391 | +0.002 |
| 涨幅（末段−首段） | +0.102 | +0.098 | — |

四段全部落在噪声内，涨幅几乎一致。判据在开跑前就写死了（后 10 步均值落在 0.36–0.42、
且前后差值同量级），不是看到结果再补的。逐步数值在 `_curve.json`。

**两条曲线可比的前提**要说清楚：v1 的 100 步 S2TT 基线是 `lora-omni-baseline @ 8bcbb42`，
而 reward 去污染那个提交 `3a6eb2f` 是**之后**为同传才做的——也就是说 v1 的 S2TT 曲线和
我们这条一样带 `<|im_end|>` 污染。所以这次不剥离恰恰是对的。将来要比同传，才必须把那个
补丁一起移过去，否则会重演 v1 记过的「BLEU 平躺 0.06、advantage≈0、学不动」。

至此 v1 → v2 的迁移在端到端层面闭环：sglang 的 `should_apply_lora` 门、Omni 的
`_lora_pattern`、Relax 的通配符展开与 PEFT 前缀、以及 adapter 传输，全部由这一次训练
隐式验证过了。

### 两条运维教训

1. **`remote()` 会把训练的生命周期绑在本地那个 modal 进程上。** 第一次跑 40 步时本地一断，
   app 在第 5 步被收掉（`Runner has been shutting down for too long`），只留下
   `iter_0000004`。改成 v1 的做法——`train.spawn(...)`，把调用交给服务端就走人，配
   `modal run --detach`。取结果用
   `modal run modal_train_s2tt.py::result --call-id <id>`。
2. **续训没生效。** 重起时本意是从 `iter_0000004` 接着跑，实际从 0 开始了，原因是上次被
   提前收掉、`latest_checkpointed_iteration.txt` 没写出来。这次反而因祸得福：拿到了完整
   的 0–39 一条曲线，和 v1 对照更干净。但 LoRA 的续训路径仍未验证过，长跑之前要单独查。
