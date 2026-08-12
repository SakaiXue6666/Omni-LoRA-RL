"""S2TT 的句级 BLEU 奖励，通过 --custom-rm-path 挂进 v2 的 Relax。

v1 是直接改 Relax 源码加了一个 rm_type=bleu 分支（`relax/engine/rewards/bleu.py`
加上 `__init__.py` 里的 elif）。v2 不需要这么干：上游有 `--custom-rm-path` 扩展点，
优先级高于 rm_type 路由，函数签名 `fn(args, sample, **kwargs)`，可以是单个 Sample
也可以是一批。所以这条 v1 delta 在 v2 里从"改框架"降级成"加一个文件"。

**算法与 v1 逐字一致，故意不做任何改进。** 因为这次跑 40 步的目的是和 v1 的
BLEU 曲线对比（v1: 前 10 步均值约 0.29，31-40 步约 0.41），reward 的定义只要动一点，
分数就没法比，也就说不清差异来自迁移还是来自奖励口径。具体沿用的三处：

  * `<answer>` 标签优先，取不到就用整串 strip；
  * label 支持裸串 / {'ground_truth': ...} / list 三种形态（S2TT 数据用 dict）；
  * sacrebleu.sentence_bleu(..., tokenize="zh")/100，夹到 [0, 1]；tokenizer 由
    metadata['tgt_lang'] 决定，缺失时按 ref 里有没有 CJK 自动判。

一处只加观测、不改分数：v1 在同传里发现过 reward 污染（sglang 返回文本带
`<|im_end|>`，拼进 response 后 BLEU 被压到真实值的约 40%）。S2TT 单轮当时没打这个
补丁，所以这里也不打，只统计有多少条响应带 `<|`，异常了能看见。
"""

from __future__ import annotations

import math
import re
from collections import Counter


try:
    import sacrebleu

    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False


ANS_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_SPECIAL = re.compile(r"<\|[^|]*\|>")

_contaminated = 0
_scored = 0


def _extract(text: str) -> str:
    m = ANS_TAG.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _ref_text(label) -> str:
    """从 label 取参考译文：支持 str / {'ground_truth': ...} / list。"""
    if isinstance(label, dict):
        ref = label.get("ground_truth") or label.get("label") or ""
    elif isinstance(label, (list, tuple)):
        ref = label[0] if label else ""
    else:
        ref = label
    return _extract(str(ref or ""))


def _sacre_tokenize(tgt_lang, ref: str) -> str:
    lang = (tgt_lang or "").lower()
    if lang in ("zh", "zh-cn") or _CJK.search(ref):
        return "zh"
    if lang == "ja":
        return "ja-mecab"
    if lang == "ko":
        return "ko-mecab"
    return "13a"


def _tokenize(s: str) -> list[str]:
    s = s.strip()
    if _CJK.search(s):
        return [ch for ch in s if not ch.isspace()]
    s = s.lower()
    s = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", s)
    return s.split()


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hyp: list[str], ref: list[str], max_n: int = 4) -> float:
    """sacrebleu 缺席时的兜底实现（带平滑，避免 log(0)）。"""
    if not hyp or not ref:
        return 0.0
    log_p = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngram_counts(hyp, n)
        ref_ng = _ngram_counts(ref, n)
        total = sum(hyp_ng.values())
        if total == 0:
            log_p.append(math.log(1.0 / (2.0 * max_n)))
            continue
        overlap = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
        p = overlap / total
        if p == 0.0:
            p = 1.0 / (2.0 * total)
        log_p.append(math.log(p))
    geo_mean = math.exp(sum(log_p) / max_n)
    hl, rl = len(hyp), len(ref)
    bp = 1.0 if hl > rl else math.exp(1.0 - rl / max(1, hl))
    return bp * geo_mean


def get_bleu_reward(response, label, metadata=None) -> float:
    """句级 BLEU ∈ [0,1]。与 v1 `relax/engine/rewards/bleu.py` 同实现。"""
    hyp_str = _extract(response)
    ref_str = _ref_text(label)
    if not hyp_str or not ref_str:
        return 0.0

    if _HAS_SACREBLEU:
        tgt_lang = (metadata or {}).get("tgt_lang") if isinstance(metadata, dict) else None
        tok = _sacre_tokenize(tgt_lang, ref_str)
        try:
            return max(0.0, min(1.0, sacrebleu.sentence_bleu(hyp_str, [ref_str], tokenize=tok).score / 100.0))
        except Exception:  # noqa: BLE001
            try:
                return max(0.0, min(1.0, sacrebleu.sentence_bleu(hyp_str, [ref_str]).score / 100.0))
            except Exception:  # noqa: BLE001
                pass

    return float(sentence_bleu(_tokenize(hyp_str), _tokenize(ref_str)))


def _score_one(sample) -> float:
    global _contaminated, _scored

    response = getattr(sample, "response", "") or ""
    _scored += 1
    if _SPECIAL.search(response):
        _contaminated += 1
        if _contaminated in (1, 10, 100) or _contaminated % 500 == 0:
            print(
                f"[bleu_rm] 响应里出现特殊 token（{_contaminated}/{_scored} 条）："
                f"{response[:120]!r} —— 不影响本次计分，但 v1 记录过这会把 BLEU 压到真实值的约四成",
                flush=True,
            )

    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    return get_bleu_reward(response, getattr(sample, "label", None), metadata=metadata)


def compute_bleu_reward(args, sample, **kwargs):
    """--custom-rm-path 的入口。单个 Sample 与一批 Sample 都要能接。"""
    if isinstance(sample, list):
        return [_score_one(s) for s in sample]
    return _score_one(sample)
