# 同传移植笔记：改了什么、哪里可能炸

同传（simultaneous S2TT）的代码从 `v1` 分支的 `Relax/examples/simul_s2tt/` 移到了本仓库的
`omni_s2tt/simul/`。

**这条路径在当前代码上一次都没跑过。** 下面把「改了哪些、为什么改、预计在哪炸、炸了长什么样」
写清楚，让第一个上手的人不用从零猜。参照指标：v1 跑 20 步是 **0.155 → 0.265**
（见 `docs/results/experiments.md` 实验 3）。

## 移了什么

| 文件 | 来源 | 说明 |
|---|---|---|
| `omni_s2tt/simul/rollout.py` | v1 同名文件 | 主体，多轮 generate（788 行） |
| `omni_s2tt/simul/audio_chunk_env.py` | 同上 | 960ms 定长切块 env，未改逻辑 |
| `omni_s2tt/simul/config.yaml` | 同上 | `max_turns: 64` / `simul_chunk_ms: 960` |
| `omni_s2tt/simul/_selftest_env.py` | 同上 | 纯 numpy 切块自测 |
| `omni_s2tt/simul/__init__.py` | 同上 | 文档字符串 |
| `omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh` | 新写 | 单轮脚本的薄包装 |

**没移** `omni_rollout.py`（396 行）—— 那是 sglang-omni Thinker 变体，当前代码已经不含
sglang-omni submodule。要用的话去 `v1` 分支拿。

## 为什么放 hub 仓而不是 Relax fork

v1 放在 `Relax/examples/simul_s2tt/`。现在放本仓库，因为 Relax fork 目前只剩 1 处
`[YULIN-MOD]`，塞进 1000 行会让跟上游 rebase 变难，也和 README 说的「换任务不需要动 Relax 的
代码」冲突。`examples.deepeyes.base_env` 的 import 仍然可用 —— Relax 根目录也在 PYTHONPATH 上。

想退回 v1 布局：把 `omni_s2tt/simul/` 挪进 `Relax/examples/simul_s2tt/`，把三处
`omni_s2tt.simul.` 改回 `examples.simul_s2tt.`，再改启动脚本的两个路径。

## 改了哪四处

| # | 位置 | 改动 | 为什么 |
|---|---|---|---|
| ① | `rollout.py` import + 638/749 行 | `_ENCODE_EXECUTOR` → `get_encode_executor()` | 当前 Relax 把那个私有全局换成了惰性构造的访问器 |
| ② | `_run_inference_step` | `post(url, payload)` → `state.post_generate(url, payload)`，函数签名多收一个 `state` | 见下面「permit」 |
| ③ | 文件末尾 | 新增 `generate.manages_inference_permit = True` | 与 ② 成对，缺一不可 |
| ④ | `generate()` | 拆成 `generate()` 外壳 + `_generate_impl()`，外壳接住 `GenerationAborted` | 上游契约要求它不能逃出去 |

**关于 permit**：当前 Relax 新增了 `relax/engine/rollout/request_permit.py`（v1 没有）。
多轮 rollout 若不声明 `manages_inference_permit`，会全程占住 session 级信号量的一个槽位，
横跨全部 ~10 轮加上中间的切块/编码时间。不死锁、不串行，是吞吐下降。上游注释原话：
*"Custom multi-turn rollouts should use inference_permit()/post_generate() instead."*

---

# 已知风险，按可能性排

## 1. 多段音频的 rope device bug —— 最可能炸的一处

**状态：代码层面逐环节核实成立，未实机复现。**

链条：

```
model.py:206   audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)   # CUDA
model.py:368   get_rope_index(..., audio_seqlens=audio_feature_lengths)           # 没 .cpu()
utils.py:9     _get_feat_extract_output_lengths()  纯算术，不改 device            # 仍 CUDA
utils.py:188   audio_len = _get_feat_extract_output_lengths(audio_seqlens[i])     # CUDA 标量
utils.py:192   st += text_len + bos_len + audio_len + eos_len                     # 污染 CPU 计数器
```

同一函数的 **video 分支显式 `.cpu()`**（`utils.py:235`、`271`），audio 分支没有 —— 和 v1
当年那个 bug 是同一形态。

**只在一条序列含 ≥2 段音频时触发。** 单轮 S2TT 每条只有 1 段，永远碰不到（所以 40 步那次一路
顺）；同传每条约 10 块，必然踩。

预期报错：

```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

栈里会出现 `get_rope_index`。**修法**（照抄 video 分支的做法，只改计算设备不改数值）：在
`model.py:368` 的调用处把 `audio_seqlens=audio_feature_lengths.cpu()`，或在 `utils.py` 进
audio 分支前 `.cpu()`。

⚠️ **v1 的补丁不能直接用。** `relax/backends/megatron/__init__.py:65` 的
`patch_qwen3_omni_rope_index_device()` patch 的是
`megatron.bridge.models.qwen_omni.modelling_qwen3_omni.model.get_rope_index`，而当前 Relax
自己 vendor 了一份在 `relax/models/qwen_omni/modeling_qwen3_omni/utils.py` —— 门牌号变了，
照搬过来等于没打。

如果确认成立，这是可以推给上游的第六个 PR，性质和已提的五个一样。

## 2. permit 声明丢了

改动 ②③ 是成对的。只改一半会直接抛：

```
RuntimeError: inference_permit()/post_generate() was called while the session-level
lock is held. A custom generate function must declare `manages_inference_permit = True`
to use per-request permits; without it ... acquiring a permit would deadlock.
```

好消息是它**大声报错而不是默默死锁**。看到这条就去检查文件末尾那行还在不在。

## 3. 双 token 流对齐 —— 最精巧、也最没验过的部分

`rollout.py` 维护两条流：`sample.rollout_tokens` 发 sglang（每段音频一个未展开 marker），
`sample.tokens` 给 Megatron（processor 已展开，与音频特征对齐），`_merge_mm_train()`
把各块的 `input_features` pad 到同一长度再沿 dim=0 拼。

这套逻辑没有针对当前版本的 processor 验证过。炸的话通常不是异常，而是**训练发散或 loss 异常**，
因为 loss_mask 和 token 对不上。要查的不变式（v1 的 CPU 自测就是断这些的）：

- `len(loss_mask) == response_length`
- `len(rollout_log_probs) == response_length`
- `len(tokens) == prompt_len + response_length`
- `sum(loss_mask) == 生成 token 总数`（不含注入的观测 token）
- sglang 调用次数 `== num_chunks`
- `input_features` 存在，且第一维 == 音频段数

`sample.metadata` 里有 `simul_num_chunks` 和 `simul_stop_reason` 可以对。

## 4. `--no-offload-train/rollout` 的 help 与代码不一致

启动脚本用了这两个 flag（v1 的常驻配置，比带 offload 快约 27~35%）。它们目前**有效** ——
`arguments.py:3170` 是 `if args.offload_train is None: = True`，只在没显式指定时才强制。

但同一处的 help 文本写着 *"This will always be true when --colocate is set."* 两者不一致。
上游哪天把代码改得和 help 一致，同传会**静默退回 offload 模式** —— 不报错，只是变慢、显存
行为变了。发现每步耗时突然从 ~3 分钟涨到 ~4.4 分钟，先查这里。

## 5. 奖励去污染的口径与单轮**不同**

| | `<\|...\|>` 特殊 token |
|---|---|
| 同传（本模块） | **去掉**再拼 response（`_clean_gen_text`，`rollout.py:156`） |
| 单轮 S2TT | **不去掉**，只报警不改分 |

这不是 bug，是各自的历史原因：

- 同传必须去 —— 每两块之间插一个 marker，跨块 2/3/4-gram 全废，实测 BLEU 被压到真实值的
  ~40%（7.2 vs 17.9）。不修的话 advantage≈0，RL 直接学不动（v1 第一次 40 步就是这么废的）。
- 单轮不去 —— 记录在案的曲线都是在带污染条件下跑的，改了就没法跟它们比。

**后果：同传曲线和单轮曲线的绝对值不可直接比较。** 各自跟自己的历史比。

## 6. 数据要求：每条样本恰好一段整段音频

`AudioChunkEnv.__init__` 有硬断言：

```python
assert len(audios) == 1, "AudioChunkEnv 期望初始恰好 1 段完整音频..."
```

切块是 env 自己做的，喂进来的必须是**整段**。单轮那份 128 条数据符合要求。v1 当时用的是
97 条整段音频，平均 9.6 秒（3.8~23.4s），960ms 一块 → 平均约 10 块一条。

---

## 没移过来的测试

v1 有两个自测，都留在 `v1` 分支的 `modal_relax_smoke.py` 里，本次没移：

- `selftest_simul`（第 2185 行）→ `selftest_simul_encode` —— **CPU，约 1 分钟**。走真实
  `process_raw_sample` 造 3 秒合成音频（4 块，末块偏短，故意复现变长特征拼接），mock 掉
  `GenerateState`/`post`，跑完整多轮 `generate()`，断上面第 3 节那些不变式。
  **这是唯一能在烧卡前发现改动 ①②③ 有没有做对的东西。**
- `selftest_rope`（第 2313 行）—— T4，1~2 分钟，纯函数复现第 1 节那个 device bug。

要用的话：`git show v1:modal_relax_smoke.py`，摘 `_offline_build_simul_out` 和
`selftest_simul_encode` 两个函数出来改写。

已经移过来的 `_selftest_env.py` 作用有限：它只测纯 numpy 的切块逻辑，用 `SimpleNamespace`
造假 sample，零 relax 依赖，而且 `examples/deepeyes/base_env.py` 在两个版本间逐字节一致 ——
所以它必然通过，验证不了任何与本次移植相关的东西。留着是因为改 chunk 逻辑时它仍然有用。

## 怎么跑

```bash
HF_CKPT=/models/qwen3-omni DATA=/s2tt/train_s2tt.jsonl NUM_ROLLOUT=20 \
SAVE_DIR=/data/s2tt/ckpt/simul_run1 \
bash omni_s2tt/run-qwen3-omni-lora-simul-4gpu.sh
```

它是单轮脚本的薄包装，超参全部复用，只加 `--custom-generate-function-path` /
`--custom-config-path` 两项，外加常驻配置。日志和 tensorboard 项目名已经和单轮分开，
不会混在一起。
