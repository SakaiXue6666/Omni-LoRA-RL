# Qwen3-Omni + LoRA RL：迁移到上游 Megatron-Bridge 0.5.0 的改造计划

> 目的：把当前"被 redai-fork `f13bec09` 逼出来的手写 direct 路线"收敛到**上游
> Megatron-Bridge 0.5.0 原生的 `export_adapter_weights` 路线**（向 miles 看齐），
> 删掉大量版本妥协代码。
>
> 铁律：**现有 direct 路线已跑通并三层验证过 LoRA 在学**，是可回退的安全基线。
> 迁移全程在 git 分支里做，每一步先隔离测试、再接线；任何一步不过就停下修正或回滚，
> 绝不带着未验证的改动往下走。

---

## 0. 背景与结论（2026-07 调研）

- **卡点根因**：镜像 pin 的 `redai-infra/megatron-bridge@f13bec09` 有 Qwen3-Omni 但**没有**
  `export_adapter_weights`；当时上游有该 API 但只有 Qwen2.5-Omni。二者不可兼得 → 手写 direct。
- **新情况**：上游 **Megatron-Bridge 0.5.0（2026-06-22）** 同时具备：
  - Qwen3-Omni 原生 bridge（HF↔Megatron 转换 + 多模态 forward）；
  - PEFT（LoRA/DoRA，按模块名挂载，模型无关）；
  - adapter 导出/流式：`export_adapter_ckpt` / `save_hf_adapter` /
    `examples/conversion/adapter/stream_adapter_weights.py`（PR #2574，含 MoE 分组专家合并与
    fused-qkv 拆分）。
- **未盖章的风险点（必须自测）**：
  1. 上游**没有 Qwen3-Omni 的 PEFT 官方 recipe/示例**（omni 官方只有全参
     `local_train_thinker_full.sh`；adapter 导出功能测试是 Qwen3 纯文本，不是 omni）；
  2. omni 已知限制：`packed_seq_params` 未实现、`inference_params` 推理未实现、自动验证仅单卡。

**决策**：值得迁移，但**先做 Phase 0 隔离验证**，通过后再动现有代码。

---

## 执行环境：Modal（所有验证/冒烟都在 Modal 上跑）

现有 Modal 资产（复用，不重造轮子）：

| 文件 | 作用 | 镜像 | GPU |
|---|---|---|---|
| `modal_verify_lora.py` | 机制层验证（attach / TP） | `slimerl/slime:latest` + 本地脚本注入 | T4 / 2×T4 |
| `modal_run.py` | sglang LoRA×RL 实验 E–J | `lmsysorg/sglang:latest` + 本地 sglang 走 PYTHONPATH 覆盖 | 单卡 |
| `modal_relax_smoke.py` | Relax 全栈 colocate 冒烟（含便宜 `probe`） | slime + redai bridge + 本地覆盖 | 4×A100 |

迁移相关的三个要点：
1. **bridge 版本只锁在一处**：`modal_relax_smoke.py` 第 66–67 行
   `pip install ... git+https://github.com/redai-infra/megatron-bridge.git@f13bec09`。
   迁移 = 复制一份镜像定义、把这行换成上游 `megatron-bridge==0.5.0`（或对应 git ref），
   **两套镜像 A/B 对跑**，随时回滚。
2. **Phase 0 脚本做成 Modal probe 函数**（沿用 `modal_verify_lora.py` 模式）：一份"上游 0.5.0
   镜像 + T4"跑完 env/attach/export/parity，几分钟低成本；全绿再上 A100。
3. **权重已缓存**在 Volume `qwen3-omni-weights`（`/models/qwen3-omni`），验证脚本直接挂载复用，
   不重下 30B。裁剪版小 checkpoint 另存，供单卡 T4/L4 跑 attach/export。

⚠️ **镜像兼容风险（Phase 0.1 首先查实）**：slime 镜像的 Megatron-LM + TE 是按特定版本编译死的；
上游 bridge 0.5.0 需要匹配的 Megatron-LM submodule + TE。"换 bridge"**可能不只是改一行 pip**——
若 0.5.0 与镜像自带 Megatron-LM 不兼容，需换基础镜像或重编译（贵、慢）。这是"能不能迁"的
第一道环境闸门。

---

## 迁移前必做：建立安全基线

- [ ] Relax、sglang 各自加 `.gitattributes`（`* text=auto eol=lf`），消除 CRLF 假 diff。
- [ ] Relax、sglang 各自 `git checkout -b lora-omni-baseline`，把当前**能跑的**改动 commit 固化。
  - 校验：`git log -1` 能看到该提交；`git status` 干净。
- [ ] 从 baseline 再切工作分支：`git checkout -b migrate-bridge-0.5`。
- [ ] （可选）用方案 A 导出 `patches/relax.patch` + `patches/sglang.patch` 存档。

> 说明：现有改动全部在工作区未提交，先固化基线是防丢的第一优先级。

---

## Phase 0 —— 隔离验证上游 0.5.0（不改现有代码，纯离线脚本）

目标：在**不碰 Relax 训练链路**的前提下，单独证明"上游 bridge 0.5.0 能给 Qwen3-Omni thinker
挂 LoRA 并正确导出 adapter"。脚本沿用 `verify_lora_*.py` 风格，**通过一个新的
`modal_migrate.py`（仿 `modal_verify_lora.py`）在 Modal T4 上跑**，几分钟低成本。

先准备"上游 0.5.0 镜像"：复制 `modal_relax_smoke.py` 的 image 定义，把 redai bridge 那行
（第 66–67 行）换成上游 `megatron-bridge==0.5.0`，其余不变。

### 0.1 环境与 API 存在性（**兼容闸门，最先跑**）
- [ ] `mig_00_env.py`（Modal T4）：
  - `import megatron.core` 与 `import megatron.bridge` 在 0.5.0 装完后**能共存**（镜像兼容关键）；
  - `from megatron.bridge import AutoBridge` 成功；
  - `hasattr(AutoBridge, "export_adapter_ckpt")` / `save_hf_adapter` 为 True；
  - `AutoBridge.from_hf_pretrained("/models/qwen3-omni", trust_remote_code=True)` 识别为 Qwen3-Omni。
- **通过标准**：全 True，provider 类名含 Omni。
- **不过则停**：若与 slime 镜像 Megatron-LM/TE 冲突 → 先解决基础镜像，否则迁移中止、维持 direct。

### 0.2 omni thinker 挂 LoRA（bridge peft，模型无关）
- [ ] 脚本 `verify/mig_01_attach.py`（可在小/裁剪 checkpoint 上跑，参考现有 `verify_lora_attach.py`）：
  - 用 `bridge.to_megatron_provider(load_weights=False)` + `provider.register_pre_wrap_hook(lora(...))`
    （miles `bridge_lora_helpers._setup_lora_model_via_bridge` 的写法）；
  - target 限定到 thinker 文本侧（`*language_model*linear_qkv/linear_proj`）。
- **通过标准**：可训参数全部是 `.adapter.`；base 全冻结；**无** adapter 落在
  `audio_tower/visual`；打印可训参数数量与现有 direct 基线一致（现基线：`192 个 / 2.556M`，rank16）。
- **对照物**：现有 `_assert_lora_attached` 的输出。

### 0.3 export_adapter → PEFT 回读验证
- [ ] 脚本 `verify/mig_02_export.py`（参考上游 `examples/conversion/adapter/verify_adapter.py`）：
  - `bridge.export_adapter_weights(model, cpu=True)` 收集 `(hf_name, tensor)`；
  - 落成 `adapter_config.json` + `adapter_model.safetensors`；
  - 用 HF `peft` 库把它加载到 HF base 上（logit 对照，能加载不报命名错即算过第一关）。
- **通过标准**：peft 能识别并加载；`target_modules` 推断正确；无命名/形状异常。

### 0.4 数值对照（关键）：bridge 导出 vs 现有 direct 手写
- [ ] 脚本 `verify/mig_03_parity.py`：对**同一份挂了 LoRA 的 omni 模型**，分别用
  1. 现有 direct：`HfWeightIteratorDirect` + `convert_qwen3omni_to_hf`（含 `_reorder_qkv_lora_b`）；
  2. 上游 bridge：`export_adapter_weights`；
  逐张量对比 **命名集合** 与 **数值**（`torch.allclose`，注意 qkv 顺序/拆分）。
- **通过标准**：两条路线产出的 `{name: tensor}` 命名一一对应、数值一致（或差异可解释，如 dtype）。
- **意义**：这是"能不能删手写代码"的判定关。**过 → bridge 路线可信，Phase 3 放心切；
  不过 → 说明 omni 的 qkv/MoE 拆分 bridge 处理与你的不一致，需先定位差异再决定。**

> Phase 0 全绿之前，不要进入 Phase 1。Phase 0 的脚本长期保留，作为回归资产。

---

## Phase 1 —— 镜像/环境切换（现有代码先不动逻辑）

- [ ] 复制 `modal_relax_smoke.py` 为 `modal_relax_smoke_v05.py`（或加环境变量开关），把 bridge 那行
      从 `redai-fork@f13bec09` 换成上游 `0.5.0`（同步 Megatron-LM/TE 依赖）。
- [ ] 先跑该文件的 `probe`（T4）确认装配（import relax / import sglang / bridge 识别 omni）。
- [ ] **非 LoRA 回归**（4×A100）：跑一个**全量权重同步**（base，不开 LoRA）的 omni smoke，确认换 bridge
      没打破既有全参链路。
- **通过标准**：probe 全绿；非 LoRA 的 rollout/train/权重同步 2 step 正常，生成连贯。
- **回滚**：旧镜像文件保留；`git checkout lora-omni-baseline` 可回旧代码+旧镜像组合。

---

## Phase 2 —— 训练侧挂 LoRA 收敛到 bridge 原生写法

把 `wrap_model_provider_with_lora`（functools.wraps 那套签名透传 hack）替换为
`provider.register_pre_wrap_hook(lora(...))`（miles 写法，bridge 官方支持的钩子）。

- [ ] 参考 miles `bridge_lora_helpers._setup_lora_model_via_bridge` + `lora_utils.create_lora_instance`
      改 `relax/backends/megatron/model.py` / `model_provider.py`。
- [ ] **保留** `_assert_lora_attached` 自检（挂载后可训参数/命名/未误挂多模态）。
- [ ] 评估删除 `wrap_model_provider_with_lora` 的签名透传 hack（坑9）——若用 pre_wrap_hook 就不需要。
- **测试**：复用 `verify/mig_01_attach.py` 的判据在**真实 Relax 启动路径**下打印，一致即过。
- **通过标准**：挂载参数数/命名与 Phase 0.2 一致；优化器参数组非空。
- **回滚**：本 phase 单独 commit，可 `git revert`。

---

## Phase 3 —— 权重同步收敛到 bridge export（删手写 direct）

> 现状：Relax 的 `hf_weight_iterator_bridge.py` **已经写了** `weight_type=="lora"` 走
> `export_adapter_weights` 的分支，但**实际同步走的是 `UpdateLoRAFromTensor`（direct 路线）**，
> 那条 bridge 分支目前是"写了但没用"。本 phase 就是把开关切到 bridge 分支。

- [ ] 让 LoRA 同步复用统一的 `update_weight_from_tensor` 的 base/lora 分流（miles 式），
      `weight_type="lora"` 走已有的 bridge 导出分支。
- [ ] **删除**：`weight_update/update_lora_from_tensor.py`（整个 `UpdateLoRAFromTensor`）、
      `weight_conversion/qwen3_omni_moe.py` 里的 `_convert_qwen3omni_lora_adapter` +
      `_reorder_qkv_lora_b`、pybase64/pickle 绕行。
- [ ] 序列化改回 `MultiprocessingSerializer`（若仍遇到 Ray authkey 问题再单独处理）。
- **测试（关键回归）**：
  - `verify/mig_03_parity.py` 已在 Phase 0 证明两路线数值一致；
  - 再加端到端：迁移后热推 adapter → sglang，`unload/load` 后生成与 direct 基线一致。
- **通过标准**：sglang 侧加载成功、生成连贯；与 baseline 分支的 rollout 输出在同 seed 下一致/可解释。
- **回滚**：本 phase 改动大，务必独立 commit；不过就 `git checkout` 回 baseline 的 direct 版本。

---

## Phase 4 —— 清理版本妥协 workaround（逐个确认后再删）

每删一项前，先确认"新版是否已内置"，删后跑对应最小测试。

- [ ] `model_provider.py` 的 `LinearCrossEntropyModule` 补注册 —— 确认 0.5.0 已内置后删。
- [ ] `common.py` / `hf_weight_iterator_direct.py` 的 `.contiguous()`（坑16）—— 若走 bridge export
      不再触及那条 all_gather 路径，可删；**保守起见先测再删**。
- [ ] `__init__.py` 的 `patch_rotary_embedding`（摘 `packed_seq_params`）—— **注意 omni 0.5.0 已知限制
      `packed_seq_params` 未实现，此 patch 很可能仍需保留**；删前务必单测。
- [ ] `arguments.py` 的 `layernorm_epsilon` 校验字段 —— 确认新版命名后再动。
- [ ] sglang 侧：锁定单一 sglang 版本后，删 `lora_manager.py` 的 `getattr(server_args,...)` 垫片。
- **通过标准**：每删一项，对应 smoke/单测不回归。

> **不要删** 的（与 bridge 版本无关，始终必要）：
> - sglang `qwen3_omni_moe.py` 的 `should_apply_lora` / `_lora_pattern`（排除 audio/vision 塔）；
> - sglang `get_audio_feature` 的不等长音频 zero-pad；
> - `patch_torch.py` 越界保护（防御性、无害）。

---

## Phase 5 —— 端到端 RL 回归（对齐迁移前后）

- [ ] 跑 `run-qwen3-30B-A3B-omni-lora-smoke.sh`（2 rollout step，colocate TP4）。
- [ ] 复现三层验证：
  - 机制层：optimizer 真更新 `lora_A/B`、base 冻结；
  - 传播层：热推 adapter 后 sglang 生成改变；
  - 效果层：翻译+BLEU，10 step reward 上升。
- **通过标准**：与 baseline 分支同口径下，reward 曲线/生成一致或更好；无崩溃。
- 通过后：合并 `migrate-bridge-0.5` → 主分支；更新镜像 pin；归档旧 direct 代码。

---

## 决策记录 / 踩坑日志（边做边补，避免忘记上下文）

> 格式：`日期 | Phase | 现象/决策/证据`

- 2026-07-07 | Phase 0 前 | 确认上游 0.5.0 具备 omni bridge + PEFT + adapter 导出三件套；
  但 omni+PEFT 无官方 recipe，需自测；`packed_seq_params` 上游未实现（patch 可能保留）。
- 2026-07-07 | Phase 0.1 | **已在 Modal(T4) 实跑 `modal run modal_migrate.py`**。结果：
  `--no-deps --force-reinstall megatron-bridge==0.5.0` 能装上（slime 自带 0.3.0rc0 → 0.5.0），
  但 **4 个闸门全部硬失败**，根因单一：
  `ImportError: cannot import name 'safe_get_world_size' from 'megatron.core._rank_utils'`
  （bridge 0.5.0 的 `utils/common_utils.py` 要求较新的 megatron.core API，slime 自带
  `/root/Megatron-LM` 的 megatron.core 太旧、没有该符号）。
  **结论**：最小改动（只换 bridge、保留 slime 的 core）不可行 → 迁移必须**连带升级
  megatron.core**（更大工程）。下一步：去掉 `--no-deps` 让 pip 拉全套依赖重测，或改用
  上游 bridge 官方镜像/指定兼容的 megatron.core commit 再跑一次 Phase 0.1。
  在此之前维持 direct 路线（现有 baseline 分支）。
- 2026-07-07 | Phase 0.1（第 2 轮）| 去掉 `--no-deps`（`pip install megatron-bridge==0.5.0`，带依赖、不 force）重跑 T4。
  **进展**：第 1 轮的 `safe_get_world_size` 报错消失 —— pip 把 megatron-core 顶到了带该 API 的版本，
  import 已越过那一关。**新阻塞（更好治）**：
  `RuntimeError: flashinfer-cubin version (0.6.8.post1) does not match flashinfer version (0.6.3)`
  —— bridge 0.5.0 把 `flashinfer-cubin` 顶到 0.6.8.post1，而 slime 自带 `flashinfer` 仍 0.6.3。
  **更严重的隐患**：运行时 `transformers` 仍是 4.57.1（bridge 要求 >=5.8.1）、`peft` 直接 `ModuleNotFoundError`
  —— 即依赖树只装了一半（`|| true` 吞掉了失败），环境不自洽。
  **结论**：用 pip 往 slime 镜像"嫁接"bridge 0.5.0 整棵依赖树**不稳**，会与 slime 钉死/编译好的栈
  （flashinfer / transformers / peft / editable megatron-core）互相打架，继续加 pip flag 是打地鼠。
  → **改走"镜像源头版本控制"**：以官方 NeMo-FW 容器（含自洽的 0.5.0 栈）为基础镜像，
  再叠 Relax/sglang/权重。见下方《镜像版本控制 / 复现策略》。
  证据佐证：NVIDIA 官方文档明说"在裸环境装这些依赖 fragile 且难复现"，官方用 Dockerfile 钉死所有依赖。
- 2026-07-08 | **决策：暂停迁移，维持 direct 路线** | 权衡后决定**不迁**到上游 0.5.0，继续用现有
  能跑的实现（slime + redai bridge `f13bec09` + 本地 Relax/sglang 覆盖，即 baseline 分支）。
  理由：Phase 0.1 两轮证明迁移的环境成本（换基础镜像、拉数十 GB NeMo 镜像、可能要 NGC 登录、
  重验证 Phase 0.2–0.4）明显高于当前收益；现有 direct 实现已可用。
  本 plan **保留存档**，若将来上游有官方 omni+PEFT recipe、或现有路线遇到硬阻塞，再从
  《镜像版本控制》方案 1（NeMo `nvcr.io/nvidia/nemo:26.06`）起第 3 轮。**在此之前不再动迁移相关代码。**

---

## 镜像版本控制 / 复现策略（Phase 0.1 两轮实测后确立）

**核心教训**：不要再用 pip 往 slime 镜像上"嫁接"bridge 0.5.0。两轮实测证明依赖树会互相打架、
且每次留下半装的不自洽环境。NVIDIA 官方文档同样明说："在裸环境装这些依赖 fragile 且难复现，
官方用 Dockerfile 钉死每一个依赖。"→ 迁移的正确起点是**换一个自洽的基础镜像**，而不是改 pip 命令。

### 基础镜像选型（三选一，优先级从上到下）

1. **官方 NeMo Framework 容器（最快、最省事，推荐先试）**
   - `nvcr.io/nvidia/nemo:26.06`（docs 标注 `0.5.0 (latest) · 26.06`，该 tag 自带自洽的 bridge 0.5.0 全栈：
     匹配的 megatron-core / TE / flashinfer / transformers / peft）。
   - 用法：把 `modal_migrate.py` 的 `MIG_BASE_IMAGE` 换成该镜像，**去掉那行 `pip install megatron-bridge`**
     （镜像里已自带），直接跑 `mig_00_env.py`。这样 Phase 0.1 才是在"官方自洽栈"上验证，而非嫁接栈。
   - 风险：NeMo 镜像很大（数十 GB）、且不含 slime/Relax 运行时；需把 Relax/sglang 以源码方式叠上（PYTHONPATH 覆盖，
     沿用 `modal_run.py` 的做法），并确认 slime 的训练入口在 NeMo 镜像里能跑。

2. **官方 `Dockerfile.ci` 自建（可完全控制、可复现，但要 build）**
   ```bash
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge megatron-bridge
   cd megatron-bridge && git checkout v0.5.0
   git submodule update --init 3rdparty/Megatron-LM   # ← Megatron-LM 版本随 submodule 钉死，正是关键
   docker build -f docker/Dockerfile.ci --target megatron_bridge -t megatron-bridge:0.5.0 .
   ```
   - 好处：megatron-core（submodule）、TE、flashinfer 全部由官方 Dockerfile 钉死，天然自洽。
   - 用于 Modal：build 后 push 到自己的 registry，再 `MIG_BASE_IMAGE=<你的registry>/megatron-bridge:0.5.0`。

3. **在 slime 镜像上补齐 + 钉死全栈（最后手段，不推荐）**
   - 只有在必须复用 slime 特定编译产物时才走。需一次性显式钉死并对齐：
     `megatron-core`、`transformer-engine`、`flashinfer` + `flashinfer-cubin` + `flashinfer-python`（三者同版本）、
     `transformers>=5.8.1,<5.9.0`、`peft>=0.18.1`、`megatron-bridge==0.5.0`，且用 `--no-cache-dir`、去掉 `|| true`
     让 build 在装失败时**硬失败**（暴露冲突，而不是留半装环境）。
   - 临时绕过 flashinfer 校验可加 `FLASHINFER_DISABLE_VERSION_CHECK=1`，但那只是掩盖症状、不解决 ABI 不匹配，
     仅用于快速判断"除 flashinfer 外是否还有别的坑"，不作为正式方案。

### 版本钉死清单（无论走哪条，最终镜像都应固化以下，写进 Dockerfile/镜像定义并记于本文件）

| 组件 | 约束来源 | 取值（迁移目标） |
|---|---|---|
| 基础镜像 | 自选 | `nvcr.io/nvidia/nemo:26.06`（方案 1）/ 自建 tag（方案 2） |
| megatron-bridge | 目标 | `0.5.0` |
| megatron-core | bridge 0.5.0 submodule | 由 `Dockerfile.ci` 的 `3rdparty/Megatron-LM` submodule 钉死（**勿用 slime 的 editable 版**） |
| transformer-engine | 镜像编译 | 跟随基础镜像（勿单独 pip 升级，避免 ABI 崩） |
| flashinfer{,-cubin,-python} | 三者必须同版本 | 跟随基础镜像；如需升级三者一起 pin 同一版本 |
| transformers | bridge 0.5.0 | `>=5.8.1,<5.9.0` |
| peft | bridge 0.5.0 | `>=0.18.1` |
| torch / CUDA | 基础镜像 | 跟随基础镜像（勿动） |

### 镜像的 A/B 与可复现要求

- **A/B 两套镜像并存、随时回滚**：A = 现有 direct 路线镜像（slime + redai bridge `f13bec09`，即 baseline）；
  B = 迁移镜像（NeMo-FW 0.5.0）。两套各自跑同一份 Phase 0.1–0.4 probe，对比数值。
- **用 digest 而非 `latest` 锁镜像**：最终确定后，把 `MIG_BASE_IMAGE` 从可变 tag 换成 `@sha256:...` digest，
  确保半年后重跑仍是同一镜像。
- **镜像定义进版本库**：Modal 的 `modal_*.py` 镜像定义（base image + pip pin + PYTHONPATH 覆盖）本身就是
  "Dockerfile"，随代码提交；每次改镜像在本文件《决策记录》记一行（日期 + 改了什么 + 为什么）。

### 下一步（Phase 0.1 第 3 轮）

- 把 `modal_migrate.py` 的 `MIG_BASE_IMAGE` 换成 `nvcr.io/nvidia/nemo:26.06`，删掉 `BRIDGE_INSTALL` 那步
  （镜像自带），T4 重跑 `mig_00_env.py`。预期：4 闸门在官方自洽栈上应能过 compat/export-api/peft，
  omni 视权重挂载而定。若过，则迁移可行，进入 Phase 0.2。
- 注意 NeMo 镜像可能需要 NGC 登录/较大拉取时间；Modal 上首次构建会慢，之后有缓存。

---

## 状态总览

| Phase | 内容 | 状态 |
|---|---|---|
| 基线 | baseline 分支已建并提交（Relax `86c97e0` / sglang `d8a1f27e7`） | ✅ 已完成（.gitattributes 待补） |
> **当前总决策（2026-07-08）：迁移暂停 ⏸️，维持 direct 路线（baseline 分支）。下表为存档，恢复迁移时再续。**

| Phase | 内容 | 状态 |
|---|---|---|
| 基线 | baseline 分支已建并提交（Relax `86c97e0` / sglang `d8a1f27e7`），**当前采用中** | ✅ 使用中 |
| 0.1 | env 兼容闸门 2 轮实跑：`--no-deps` 不可行（core 太旧）；带依赖 pip 嫁接也不可行（flashinfer 冲突 + transformers/peft 半装）。**决策：暂停迁移** | ⏸️ 暂停（见踩坑日志） |
| 0.2–0.4 | attach / export / parity 验证脚本 | ⏸️ 暂停 |
| 1 | 镜像切换 + 非 LoRA 回归 | ⏸️ 暂停 |
| 2 | 训练侧挂 LoRA 收敛 bridge pre_wrap_hook | ⏸️ 暂停 |
| 3 | 权重同步切 bridge export，删 direct 手写 | ⏸️ 暂停 |
| 4 | 清理版本妥协 workaround | ⏸️ 暂停 |
| 5 | 端到端 RL 回归 + 合并 | ⏸️ 暂停 |

图例：⬜ 未开始 / 🟡 进行中 / ✅ 通过 / ⏸️ 暂停 / ❌ 卡住（见日志）
