# v1 冻结存档（direct 路线）

本文件冻结 **v1** —— 2026-04 至 2026-07 期间跑通 Qwen3-Omni Thinker + LoRA 强化学习的那一套
「代码 + 镜像」组合。v2 迁移到新版 Relax 官方 LoRA adapter 路线后，v1 不再演进，但必须保持
**可复现**，因为它是 v2 唯一的 parity oracle（对照基准）：v2 的 LoRA 挂载范围、adapter 导出
命名、reward 曲线，都要和 v1 对齐才算迁对。

> 逐行 diff 以 git 为准，改动的分类与解释见 `IMPORTANT/DIFF_SUMMARY.md`，
> 踩坑细节见 `IMPORTANT/LORA_RL_INTEGRATION.md`，迁移评估见 `IMPORTANT/MIGRATION_PLAN.md`。

---

## 0. 为什么需要这份文档：v1 当前**并不**可复现

所有 v1 的 Modal 脚本写的都是可变 tag：

```python
BASE_IMAGE = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")
```

而 `slimerl/slime:latest` 一直在移动 —— 2026-08-10 查到的 `latest` 构建于当天，
`transformers` 已是 5.12.x、`megatron-bridge` 已是 0.5.0。**v1 之所以到 7 月还能跑，
靠的是 Modal 的镜像缓存**：Modal 按「镜像定义字符串」缓存，
`from_registry("slimerl/slime:latest")` 只要定义不变就复用最初构建的那一份，不重新解析 tag。

这份缓存不是永久的。**一旦 Modal 缓存过期，同一行代码会拉到当天的镜像，v1 立刻起不来。**
所以第 1 节的 digest 必须写进代码，替换掉 `:latest`。

---

## 1. 镜像

### 1.1 身份

```
slimerl/slime:nightly-dev-20260428a
slimerl/slime@sha256:bd219aba21be6e404ff09e385f34f40993b60773b928e13f341e8d77590da6aa
```

**认定依据**（`latest` 不可回溯，只能取证反查）：

1. 用 v1 原本那行镜像定义命中 Modal 缓存，dump 出真实运行时版本（`modal_v1_probe.py` /
   `modal_v1_deep.py`）。其中 `transformers 4.57.1` + `flashinfer 0.6.3` 与
   `IMPORTANT/MIGRATION_PLAN.md` 里 2026-07-07 那轮记录完全吻合，确认读到的就是 v1 环境。
2. 镜像内部有**两层时间戳**，必须分清（我一开始就在这里判断错过一次）：
   - `2026-02-23 12:03~04 UTC`：torch / sglang / transformers / flashinfer
     —— 这是**基础镜像** `lmsysorg/sglang` 的层，**不代表 slime 镜像的构建时间**。
   - `2026-04-27~28 UTC`：ray / megatron-core / megatron-bridge / transformer_engine /
     sglang_router / `slime==0.2.4`(editable) —— 这才是 **slime 自己的构建层**。
3. 决定性证据：`ray 2.55.1` 在 PyPI 的发布日期是 **2026-04-22**，所以镜像必然构建于此之后，
   二三月的 nightly 全部排除（实测 `20260225a` / `0226a` / `0227a` 的 ray 均为 2.54.0，
   其余 10 个包全一致）。
4. 缓存镜像里 mtime 最新的文件是 `2026-04-28 06:11:35 UTC`；
   `nightly-dev-20260428a` 的 config `created` = `2026-04-28T14:11:35.58+08:00`
   = **06:11:35 UTC**，精确到秒相同。
5. 终验：拉该 digest 与 v1 缓存镜像逐包比对，**11/11 全部一致**（含 ray 2.55.1）。

### 1.2 版本表（实测）

| 包 | 版本 | 来源层 |
|---|---|---|
| python | 3.12.3 (main, Jan 22 2026) | base |
| torch | 2.9.1+cu129 | base 2/23 |
| transformers | 4.57.1 | base 2/23 |
| **peft** | **未安装** | — |
| sglang | 0.5.9（运行时被本地 fork 经 PYTHONPATH 覆盖） | base 2/23 |
| sgl-kernel | 0.3.21 | base 2/23 |
| flashinfer-python / -cubin | 0.6.3 | base 2/23 |
| flashinfer-jit-cache | 0.6.3+cu129 | base 2/23 |
| megatron-core | 0.16.0rc0 | slime 4/27 |
| **megatron-bridge** | **0.3.0rc0** | slime 4/27 |
| transformer-engine | 2.10.0 | slime 4/27 |
| ray | 2.55.1 | slime 4/27 |
| slime | 0.2.4（editable） | slime 4/28 |
| numpy | 1.26.4 | base |
| flash_attn | 2.7.4.post1 | base |

镜像内共 348 个发行包，完整清单用 `modal run modal_v1_deep.py` 随时重新导出。

**两个关键含义**（决定了 v1 为什么长成现在这样）：

- `megatron-bridge` 是 **0.3.0rc0**，不是 0.5.0。v1 的手写 direct 导出
  (`update_lora_from_tensor.py`) 建在 0.3.0rc0 的行为上，做 v2 export parity 时
  **不能假设两边 API 语义一致**。
- `peft` **根本没装**。这解释了 v1 为什么走自己写的 `apply_lora_to_model` 而不是
  bridge PEFT attach —— 当时环境里没有这个选项。

### 1.3 镜像之上叠加的东西

v1 不是裸用 slime 镜像，`modal_relax_smoke.py` 在 build 阶段还叠了：

- Relax 的 `requirements.txt` + `tensordict==0.10.0` + `pyvers==0.1.0`
- **redai fork 的 megatron-bridge**：`git+https://github.com/redai-infra/megatron-bridge.git@f13bec09`
  （`--no-deps --force-reinstall`，`Qwen3OmniMoEBridge` 在这里；它**没有**
  `export_adapter_weights`，这正是 v1 必须手写 direct 导出的根因）
- `transferqueue`（redai fork）、`sacrebleu`（S2TT 的 BLEU reward）
- 本地 `sglang` fork 经 **PYTHONPATH 前置**覆盖镜像自带的 sglang

因为用了 `--no-deps`，redai bridge 不会改动 transformers/flashinfer 版本 ——
所以 1.2 那张表就是 base 镜像自带的版本。

---

## 2. 代码冻结

hub 仓 `omni-lora-rl`（`https://github.com/SakaiXue6666/omni-lora-rl.git`）：

| 项 | 值 |
|---|---|
| 分支 | `sglang_omni` |
| commit | `61136dde` (2026-07-24T23:55+08) `docs: document Omni integration changes` |

三个 fork（均在 `lora-omni-baseline` 分支，工作区干净）：

| submodule | fork | commit | 上游基线（见 DIFF_SUMMARY §1） |
|---|---|---|---|
| `Relax/` | `SakaiXue6666/Relax` | `6cde0798` (2026-07-24T23:53) | `redai-infra/Relax` @ `01973f3` |
| `sglang/` | `SakaiXue6666/sglang` | `d13903a9` (2026-07-24T23:53) | `sgl-project/sglang` @ `19b60a4f9` |
| `sglang-omni/` | `SakaiXue6666/sglang-omni` | `185d6526` (2026-07-24T23:53) | `sgl-project/sglang-omni` @ `5cefa39e` |

三个上游基线均由 `git merge-base HEAD origin/main` 实测复核过。相对基线的完整改动已导出为
兜底 patch，见 `patches/`（即使 fork 仓库丢失也能还原）：

| patch | 规模 |
|---|---|
| `patches/relax.patch` | 33 文件 +4723/-61 |
| `patches/sglang.patch` | 7 文件 +652/-16 |
| `patches/sglang-omni.patch` | 26 文件 +2187/-55 |

> `IMPORTANT/DIFF_SUMMARY.md` 里记的快照是 `Relax 86c97e0` / `sglang d8a1f27e7`，
> 那是写该文档当时的快照；此后分支又往前走到上表的 commit（主要是 sglang-omni 接入与文档）。
> **以上表为 v1 的最终冻结点。**

---

## 3. v1 实现了什么

分类与逐文件说明见 `IMPORTANT/DIFF_SUMMARY.md`，这里只留骨架，便于 v2 对照检查。

> 注意 `DIFF_SUMMARY.md` 记的文件数/行数（Relax 17 文件 +1001/-41、sglang 7 文件 +554/-16）
> 对应的是更早的快照 `86c97e0` / `d8a1f27e7`；本冻结点已增长到 Relax 33 文件 +4723/-61、
> sglang 7 文件 +652/-16，另加 sglang-omni 26 文件 +2187/-55（增量主要是 sglang-omni 接入
> 与 Talker/语音链路）。分类结论仍适用，行数以 `patches/` 为准。

**训练侧（Relax）**

- LoRA 挂载：`model_provider.py` 的 `apply_lora_to_model` / `wrap_model_provider_with_lora`（自己挂，不走 PEFT）
- 权重同步：`update_lora_from_tensor.py` 手写把 adapter 张量从 Megatron 推到 sglang
- Omni 专属命名转换：`weight_conversion/qwen3_omni_moe.py` 的 adapter 改名 + qkv 重排
- 任务相关：BLEU reward、simul S2TT 多轮 rollout、启动脚本

**推理侧（sglang，7 文件 +554/-16）**

- `models/qwen3_omni_moe.py`：`should_apply_lora()` 把 audio/vision tower 排除在 LoRA 之外，
  只挂 thinker；audio feature padding 修复
- `lora/lora_manager.py`：接 `should_apply_lora` 钩子
- TP 权重更新修复：`tp_worker.py` 的 `monkey_patch_torch_reductions()`、`patch_torch.py` 的边界检查

**v2 迁移时的对照重点**：上游 sglang 至今仍没有任何 Omni LoRA 逻辑，所以
`should_apply_lora` 那一路是**永久 delta**，不是待还的技术债；而手写 direct 导出
是版本妥协的产物，v2 应由 bridge 的 `export_adapter_weights` 取代。

---

## 4. 复现 v1

```bash
git clone --recursive https://github.com/SakaiXue6666/omni-lora-rl.git
cd omni-lora-rl
git checkout 61136dde
git submodule update --init --recursive
```

Modal 侧把镜像钉成 digest（替换 `:latest`）：

```powershell
$env:RELAX_BASE_IMAGE = "slimerl/slime@sha256:bd219aba21be6e404ff09e385f34f40993b60773b928e13f341e8d77590da6aa"
modal run modal_relax_smoke.py
```

---

## 5. 冻结动作清单

- [x] 定位并认定 v1 镜像 digest（第 1.1 节）
- [x] 把 8 个 v1 Modal 脚本的镜像默认值改成该 digest：`modal_relax_smoke.py`、
      `modal_relax_smoke.baseline.py`、`modal_verify_lora.py`、`modal_omni_serve_lora.py`、
      `modal_omni_serve_lora_tensor.py`、`modal_omni_serve_speech.py`、
      `modal_test_sglang_omni.py`、`modal_migrate.py`
      （`RELAX_BASE_IMAGE` / `MIG_BASE_IMAGE` 环境变量仍可覆盖；
      `modal_v1_probe.py` / `modal_v1_deep.py` 保留 `:latest`，它们的作用就是探缓存）
- [x] 导出存档 patch：`patches/relax.patch`、`patches/sglang.patch`、`patches/sglang-omni.patch`
      （即使 fork 仓消失也能还原；三个均通过 `git apply --check --reverse` 校验，用法见 `patches/README.md`）
- [ ] 给 hub 仓与三个 fork 打 `v1-frozen` tag
- [ ] v2 在独立分支/目录开工，不覆盖 v1

---

## 6. 取证脚本

| 脚本 | 作用 |
|---|---|
| `modal_v1_probe.py` | 走 Modal 缓存 dump v1 镜像的关键包版本 |
| `modal_v1_deep.py` | 全量包清单 + dist-info mtime 分层 + 安装来源取证（认定镜像的关键） |
| `slime_tag_lookup.py` | 本地查 Docker Registry 元数据，按 config `created` 反查 tag + digest，不拉镜像 |
| `modal_v1_confirm.py` | 拉候选镜像与 v1 版本表逐包比对，`V1_CANDIDATE` 环境变量切候选 |

---

## 7. 时间线

| 日期 | 事件 |
|---|---|
| 2026-02-23 | 基础镜像 `lmsysorg/sglang` 构建（torch 2.9.1 / sglang 0.5.9 / transformers 4.57.1） |
| 2026-04-28 | slime `nightly-dev-20260428a` 构建；Modal 缓存下这一份 `latest`，此后一直复用 |
| 2026-07-07 | Phase 0.1 两轮尝试 pip 嫁接 megatron-bridge 0.5.0，均失败（core 太旧 / flashinfer 冲突 + transformers、peft 半装） |
| 2026-07-08 | 决策：暂停迁移，维持 direct 路线 |
| 2026-07-24 | v1 最终快照（hub `61136dde` + 三个 fork） |
| 2026-08-10 | 新版 Relax 官方镜像上环境闸门全部通过，v2 迁移成立；回溯认定 v1 镜像 digest |
