# Qwen3-Omni Talker Prefill / Decode 流程笔记

## 符号约定

```text
id(输入thinker文本)       = thinker 看到的文本 token id
id(输入thinker音频)       = thinker 看到的音频占位 / 音频 token id
id(输出thinker文本)       = thinker 生成出来的 assistant 文本 token id

e(x)                    = x 的最底层 embedding / 最底层语义
h_thinker(x)[i]          = thinker 第 i 层 hidden state
talker_h(x)[i]           = talker 第 i 层 hidden state

c1, c2, c3, ... cN       = 一帧音频的 N 个码本
id(c1_t)                = 第 t 帧，第 1 个码本的 code id
e(c1_t)                 = 第 t 帧，第 1 个码本 code 的 embedding
```

这里 `c1` 可以理解成主码本，也就是代码里的：

```python
input_ids[:, -1:]
```

`c2...cN` 是 `code_predictor` 补出来的 residual codes。

## 1. Thinker 端流程

先看 thinker 自己在做什么。假设用户输入里有文本和音频。

原始输入可以记成：

```text
输入thinker文本
输入thinker音频
```

进入 thinker 之后，底层表示大概是：

```text
Thinker prefill 输入：
[
  e(输入thinker文本),
  e(输入thinker音频)
]
```

如果输入里有音频，音频不是简单的普通文本 token。它通常先经过音频 encoder / projector，变成 thinker 能吃的 embedding，所以这里简写成：

```text
e(输入thinker音频)
```

thinker 的 prefill 会一次性读完整个输入上下文：

```text
[
  e(输入thinker文本),
  e(输入thinker音频)
]
```

然后建立 KV cache，并在最后一个位置预测 assistant 输出文本的第一个 token：

```text
Thinker prefill 输出：
id(输出thinker文本_0)
```

接下来 thinker 进入 decode，每一步只吃上一步自己生成的文本 token：

```text
Thinker decode 第 0 步输入：
[
  e(输出thinker文本_0)
]

Thinker decode 第 0 步输出：
id(输出thinker文本_1)
```

再下一步：

```text
Thinker decode 第 1 步输入：
[
  e(输出thinker文本_1)
]

Thinker decode 第 1 步输出：
id(输出thinker文本_2)
```

所以 thinker 端完整输出文本可以记成：

```text
id(输出thinker文本) = [
  id(输出thinker文本_0),
  id(输出thinker文本_1),
  id(输出thinker文本_2),
  ...
]
```

同时，thinker 在 prefill 和 decode 过程中都会产生 hidden states：

```text
h_thinker(输入thinker文本)[0..L]
h_thinker(输入thinker音频)[0..L]
h_thinker(输出thinker文本_0)[0..L]
h_thinker(输出thinker文本_1)[0..L]
h_thinker(输出thinker文本_2)[0..L]
...
```

合起来可以粗略写成：

```text
h_thinker(输出thinker文本)[0..L]
```

其中某一层，例如：

```text
h_thinker(输入thinker音频)[accept_hidden_layer]
```

会被拿去给 talker 用，因为音频 / 图像这类多模态输入不是普通文本 embedding，talker 需要 thinker 已经理解过的 hidden state。

thinker 生成出来的 assistant 文本，也会被拿去给 talker 当朗读内容：

```text
id(输出thinker文本)
e(输出thinker文本)
h_thinker(输出thinker文本)[0..L]
```

简单说，thinker 端给 talker 的东西主要有两类：

```text
1. 要朗读的文本：
[
  id(输出thinker文本),
  e(输出thinker文本)
]

2. 已经理解过的多模态信息：
[
  h_thinker(输入thinker音频)[accept_hidden_layer]
]
```

## 2. 组装 Talker 的 Prefill 输入

Talker 的 prefill 输入大概可以写成：

```text
输入talker文本/上下文 = [
  e(输入thinker文本),
  h_thinker(输入thinker音频)[accept_hidden_layer],
  e(输出thinker文本的前几个特殊token),
  e(tts_bos),
  e(输出thinker文本第一个token)
]
```

更贴近代码一点：

```text
talker_prefill_inputs = [
  text_projection(e(用户文本)),
  hidden_projection(h_thinker(用户音频)[accept_hidden_layer]),
  text_projection(e(输出thinker文本前缀)),
  e(tts_bos),
  text_projection(e(输出thinker文本第一个字))
]
```

然后还有一个单独保存的东西：

```text
trailing_text_hidden = [
  text_projection(e(输出thinker文本第2个字)),
  text_projection(e(输出thinker文本第3个字)),
  text_projection(e(输出thinker文本第4个字)),
  ...,
  e(tts_eos)
]
```

也就是说：

```text
输入talker文本的一部分先放进 prefill
剩下的输出thinker文本 hidden 放进 trailing_text_hidden
```

## 3. Talker Prefill 做什么

prefill 时，talker 一次性吃进去：

```text
talker_prefill_inputs
```

然后输出第 0 帧音频的第一个码本：

```text
输出：id(c1_0)
```

注意：这里只生成了：

```text
id(c1_0)
```

还没有完整的一帧音频，因为完整一帧需要：

```text
[id(c1_0), id(c2_0), id(c3_0), ..., id(cN_0)]
```

## 4. Talker Decode 第 0 步

现在 `prepare_inputs_for_generation` 里的 decode 逻辑开始工作。

已有：

```text
上一轮输出 = id(c1_0)
```

先取它的 embedding：

```text
e(c1_0)
```

再取 talker 上一步最后一层 hidden：

```text
talker_h(c1_0)[last]
```

然后送进 `code_predictor`：

```text
code_predictor 输入：
[
  talker_h(c1_0)[last],
  e(c1_0)
]
```

`code_predictor` 输出剩余码本：

```text
输出：
id(c2_0), id(c3_0), ..., id(cN_0)
```

于是第 0 帧完整音频码就是：

```text
第0帧音频码 = [
  id(c1_0),
  id(c2_0),
  id(c3_0),
  ...,
  id(cN_0)
]
```

## 5. 把第 0 帧变成下一步输入

有了完整一帧之后，把每个码本 embedding 加起来：

```text
e(第0帧音频) =
  e(c1_0)
+ e(c2_0)
+ e(c3_0)
+ ...
+ e(cN_0)
```

然后再加上当前对应的文本 hidden。

如果当前是第 0 个文本位置：

```text
下一步 talker 输入 =
  e(第0帧音频)
+ trailing_text_hidden[0]
```

也就是：

```text
下一步 talker 输入 =
  [
    e(c1_0) + e(c2_0) + ... + e(cN_0)
  ]
  +
  text_projection(e(输出thinker文本第2个字))
```

然后 talker 用这个输入继续 forward，输出下一帧的主码本：

```text
输出：id(c1_1)
```

## 6. Decode 第 1 步

同理。

已有：

```text
id(c1_1)
```

`code_predictor` 输入：

```text
[
  talker_h(c1_1)[last],
  e(c1_1)
]
```

`code_predictor` 输出：

```text
id(c2_1), id(c3_1), ..., id(cN_1)
```

第 1 帧完整音频码：

```text
第1帧音频码 = [
  id(c1_1),
  id(c2_1),
  id(c3_1),
  ...,
  id(cN_1)
]
```

下一步输入：

```text
下一步 talker 输入 =
  e(c1_1) + e(c2_1) + ... + e(cN_1)
  + trailing_text_hidden[1]
```

然后输出：

```text
id(c1_2)
```

## 7. 最简总流程

可以把整个过程记成这个：

```text
Thinker 输入：
[
  e(输入thinker文本),
  e(输入thinker音频)
]

Thinker 输出：
[
  id(输出thinker文本),
  h_thinker(输入thinker音频)[accept_hidden_layer],
  e(输出thinker文本)
]

Talker prefill 输入：
[
  e(输入talker文本),
  h_thinker(输入thinker音频)[accept_hidden_layer],
  e(输出thinker文本开头)
]

Talker prefill 输出：
id(c1_0)

Talker decode 第 t 步：

code_predictor 输入：
[
  talker_h(c1_t)[last],
  e(c1_t)
]

code_predictor 输出：
[
  id(c2_t),
  id(c3_t),
  ...,
  id(cN_t)
]

完整第 t 帧音频码：
[
  id(c1_t),
  id(c2_t),
  id(c3_t),
  ...,
  id(cN_t)
]

下一步 talker 输入：
[
  e(c1_t) + e(c2_t) + ... + e(cN_t)
  + trailing_text_hidden[t]
]

Talker 输出：
id(c1_{t+1})
```

## 8. 为什么 Prefill 和 Decode 输入不一样

prefill 和 decode 输入长得不一样，是因为它们的任务不同：

```text
prefill 阶段还没有历史音频帧，
所以只能输入文本 / 多模态上下文，
用来预测第一帧主码本 id(c1_0)。
```

```text
decode 阶段已经有上一帧完整音频，
所以输入变成：

上一帧完整音频 embedding + 当前文本 hidden

也就是：

e(c1_t) + e(c2_t) + ... + e(cN_t) + trailing_text_hidden[t]

用来预测下一帧主码本 id(c1_{t+1})。
```

对应关系：

```text
prefill:
[
  条件上下文
]
→ id(c1_0)

decode:
[
  第 t 帧完整音频
  + 第 t 个文本 hidden
]
→ id(c1_{t+1})
```
