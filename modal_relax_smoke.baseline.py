"""在 Modal 上跑 Relax 的 Qwen3-Omni thinker-LoRA「启动冒烟」（colocate 4 卡）。

目的：把 Relax 全栈拉起来，验证 Block 3/4/5 的 LoRA 链路在真实运行时能跑通
（不看训练效果，跑 2 个 rollout step 就停）。对应启动脚本：
    Relax/scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh

────────────────────────────────────────────────────────────────────────────
【镜像策略】没有现成的 Relax 镜像，所以基于能拉到的 slime 预构建镜像
`slimerl/slime:latest`（已含 Megatron-LM + Megatron-Bridge + TransformerEngine
+ flash-attn + ray），在 build 阶段补上 Relax 的 python 依赖 + redai fork 的
megatron-bridge（Qwen3OmniMoEBridge 在这里），并把【本地这份】sglang（含 Block
1/2 的 Qwen3-Omni LoRA 改动）通过 PYTHONPATH 注入到最前面覆盖镜像自带的 sglang。

⚠️ 这套拼装有三个无法在本地验证、必须先在 Modal 上查实的点（见 probe）：
   1. slime 镜像的 megatron.bridge 是否 redai fork（认得 Qwen3-Omni）；
      若不是，build 阶段的 force-reinstall 会换成 redai fork。
   2. 本地 sglang 覆盖后 sgl-kernel/flashinfer 二进制是否对得上（能否 import sglang）。
   3. 装完 Relax requirements 后能否 import relax（依赖/版本冲突）。

用法（Windows 控制台先切 UTF-8）：
    chcp 65001; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    modal run modal_relax_smoke.py::probe     # 先跑便宜探针（T4，几分钟）
    modal run modal_relax_smoke.py            # 探针通过后再上 4×A100 冒烟
"""

from __future__ import annotations

import os
import pathlib

import modal

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
RELAX_LOCAL = HERE / "Relax"                     # 含 Block 3/4/5 改动的本地源码
SGLANG_LOCAL = HERE / "sglang" / "python"        # 本地 sglang（Block 1/2 改动）

# 镜像里的落点
RELAX_REMOTE = "/root/Relax"
SGLANG_REMOTE = "/root/sglang_src/python"        # 放 PYTHONPATH 最前，覆盖镜像自带 sglang

# 模型权重 Volume（modal_run.py 已把 Qwen3-Omni-30B 下到这里，直接复用）
MODEL_VOLUME_NAME = "qwen3-omni-weights"
MODEL_DIR = "/models/qwen3-omni"                 # Volume 内的权重目录

# PYTHONPATH：本地 sglang 最前 -> Relax -> （megatron 用镜像安装的）
MEGATRON_REMOTE = "/root/Megatron-LM"
PYTHONPATH = f"{SGLANG_REMOTE}:{RELAX_REMOTE}:{MEGATRON_REMOTE}"

# 纯文本冒烟（最快路径，不喂 image/audio）。设 "0" 则走多模态（需准备真实数据）。
TEXT_ONLY = os.environ.get("TEXT_ONLY", "1") == "1"

BASE_IMAGE = os.environ.get("RELAX_BASE_IMAGE", "slimerl/slime:latest")


image = (
    modal.Image.from_registry(BASE_IMAGE, add_python=None)
    .add_local_file((RELAX_LOCAL / "requirements.txt").as_posix(), "/tmp/relax-req.txt", copy=True)
    .run_commands(
        # Relax 的 python 依赖（轻量；重编译依赖由 slime 镜像提供）
        "pip install --no-cache-dir -r /tmp/relax-req.txt || true",
        "pip install --no-cache-dir tensordict==0.10.0 pyvers==0.1.0 --no-deps || true",
        # redai fork 的 megatron-bridge：Qwen3OmniMoEBridge 在这里
        "pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall "
        "git+https://github.com/redai-infra/megatron-bridge.git@f13bec09 || true",
        "pip install --no-cache-dir --no-deps "
        "'transferqueue @ git+https://github.com/redai-infra/TransferQueue.git' || true",
        # S2TT reward 用 sacrebleu(中文 zh tokenizer)，对齐 my_omni quality_reward
        "pip install --no-cache-dir sacrebleu || true",
    )
    .add_local_dir(
        RELAX_LOCAL.as_posix(),
        RELAX_REMOTE,
        copy=True,
        ignore=["**/.git", "**/__pycache__", "**/*.pyc", "**/docs", "**/node_modules"],
    )
    .add_local_dir(
        SGLANG_LOCAL.as_posix(),
        SGLANG_REMOTE,
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_file((HERE / "verify_lora_e2e.py").as_posix(), "/root/verify_lora_e2e.py", copy=True)
    .add_local_file((HERE / "verify_lora_learning.py").as_posix(), "/root/verify_lora_learning.py", copy=True)
    .env({"PYTHONPATH": PYTHONPATH, "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

app = modal.App("relax-omni-lora-smoke")
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
# S2TT(英语音频→中文)数据卷：FLEURS 下载的 wav + jsonl 持久化，跨 run 复用，避免重复下载。
s2tt_volume = modal.Volume.from_name("s2tt-data", create_if_missing=True)
S2TT_DIR = "/s2tt"
S2TT_JSONL = f"{S2TT_DIR}/train_s2tt.jsonl"

# 下载 FLEURS 用的轻量镜像（datasets 必须 <3，否则 google/fleurs 脚本数据集加载被禁）。
prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("datasets>=2.19,<3", "numpy<2", "soundfile", "librosa", "huggingface_hub<0.26")
)


@app.function(
    image=prep_image,
    volumes={S2TT_DIR: s2tt_volume},
    timeout=60 * 60,
)
def prep_s2tt(limit: int = 128, split: str = "validation") -> None:
    """下载 FLEURS en→zh S2TT，写 wav + Relax 风格 jsonl 到 s2tt 卷。

    - en_us 提供音频，cmn_hans_cn 按 id 对齐提供中文参考（N-way 平行）。
    - 输出 jsonl 每行：prompt(含 <audio> 占位) / audios(wav 绝对路径) /
      label({ground_truth}) / metadata({src_lang,tgt_lang,rm_type:bleu})。
    """
    import json
    import os
    import wave

    import numpy as np
    from datasets import load_dataset

    audio_dir = f"{S2TT_DIR}/audio"
    cache_dir = f"{S2TT_DIR}/.hf-cache"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    def _write_wav(path, arr, sr):
        a = (np.asarray(arr, dtype="float32").clip(-1, 1) * 32767).astype("int16")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(a.tobytes())

    print(f"[prep] loading tgt cmn_hans_cn[{split}] text ...", flush=True)
    tgt_by_id = {}
    ds_tgt = load_dataset("google/fleurs", "cmn_hans_cn", split=split,
                          cache_dir=cache_dir, trust_remote_code=True)
    for x in ds_tgt.select_columns(["id", "transcription"]):
        t = (x.get("transcription") or "").strip()
        if t:
            tgt_by_id.setdefault(x["id"], t)
    print(f"[prep] tgt records: {len(tgt_by_id)}", flush=True)

    print(f"[prep] streaming src en_us[{split}] + join ...", flush=True)
    ds_src = load_dataset("google/fleurs", "en_us", split=split,
                          cache_dir=cache_dir, trust_remote_code=True)
    prompt = ("<audio>\nPlease translate the English speech into Chinese. "
              "Only output the Chinese translation.")
    n = 0
    with open(S2TT_JSONL, "w", encoding="utf-8") as fout:
        for row in ds_src:
            cid = row["id"]
            ref = tgt_by_id.get(cid)
            if not ref:
                continue
            audio = row["audio"]
            wav_path = f"{audio_dir}/fleurs_{cid:08d}_en.wav"
            try:
                _write_wav(wav_path, audio["array"], audio.get("sampling_rate", 16000))
            except Exception as e:  # noqa: BLE001
                print(f"[prep] skip id={cid}: {e}", flush=True)
                continue
            rec = {
                "prompt": prompt,
                "audios": [wav_path],
                "label": {"ground_truth": ref},
                "metadata": {
                    "src_lang": "en", "tgt_lang": "zh", "rm_type": "bleu",
                    "src_text": (row.get("transcription") or "").strip(),
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"[prep] wrote {n} ...", flush=True)
            if limit and n >= limit:
                break

    s2tt_volume.commit()
    print(f"[prep] DONE: wrote {n} records -> {S2TT_JSONL}", flush=True)


@app.local_entrypoint()
def prep(limit: int = 128, split: str = "validation") -> None:
    prep_s2tt.remote(limit=limit, split=split)


# ---------------------------------------------------------------------------
# 探针：便宜地查实镜像层面的三个不确定点（T4，几分钟，不烧 4 卡）
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="T4", volumes={"/models": model_volume}, timeout=20 * 60)
def run_probe() -> None:
    import subprocess
    import sys

    script = r'''
import os, importlib
print("="*60)
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())

# 1) 本地 sglang 是否解析到我们注入的源码 + sgl-kernel 二进制是否兼容
try:
    import sglang
    base = os.path.dirname(sglang.__file__)
    print("[sglang] from", base, "| local-override?", base.startswith("/root/sglang_src"))
    from sglang.srt.lora.lora_manager import LoRAManager  # 触发 LoRA 相关 import
    print("[sglang] LoRAManager import OK (sgl-kernel 兼容)")
except Exception as e:
    print("[sglang] FAIL:", repr(e))

# 2) megatron.bridge 是否 redai fork（认得 Qwen3-Omni）
try:
    import megatron.core
    print("[megatron.core] from", os.path.dirname(megatron.core.__file__))
    import megatron.bridge as mb
    print("[megatron.bridge] from", os.path.dirname(mb.__file__))
    from megatron.bridge import AutoBridge
    try:
        br = AutoBridge.from_hf_pretrained("/models/qwen3-omni", trust_remote_code=True)
        print("[AutoBridge] 识别 Qwen3-Omni OK ->", type(br).__name__)
    except Exception as e:
        print("[AutoBridge] from_hf_pretrained FAIL:", repr(e)[:300])
except Exception as e:
    print("[megatron.bridge] FAIL:", repr(e))

# 3) 能否 import relax
try:
    import relax
    print("[relax] import OK from", os.path.dirname(relax.__file__))
except Exception as e:
    print("[relax] FAIL:", repr(e)[:300])
print("="*60)
'''
    proc = subprocess.run([sys.executable, "-c", script], check=False)
    raise SystemExit(proc.returncode)


# ---------------------------------------------------------------------------
# 接线层自测（纯 CPU，秒级）：不加载 30B / 不起 sglang / 不起 Ray，
# 只验最容易在「真烧 4 卡时才暴露」的接线问题——provider 包装层的签名派发。
# 对应 LORA_RL_INTEGRATION.md 坑 8/9。
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="T4", timeout=15 * 60)
def verify_learning() -> None:
    """机制层验证：LoRA 真在学吗？

    不加载真实 30B / 不起 sglang / 不跑 RL，用真实 Megatron 模块搭最小树挂 LoRA，
    跑几步 forward+backward+step，断言 adapter 更新、base 冻结不动。
    对应 verify_lora_learning.py，T4 单卡几分钟。
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-u", "/root/verify_lora_learning.py"],
        check=False,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    raise SystemExit(proc.returncode)


@app.function(image=image, timeout=10 * 60)
def verify_wiring() -> None:
    import subprocess
    import sys

    script = r'''
import inspect
import relax.backends.megatron.model_provider as mp

# 把真正挂 LoRA 的步骤打桩成 no-op：本测只关心「包装层是否破坏 Megatron 的
# 按签名传参」，不需要真的建模型/挂 adapter。
mp.apply_lora_to_model = lambda model, args: None
wrap = mp.wrap_model_provider_with_lora

failures = []

# 模拟 bridge 模式：original_provider 是 ModelProvider.provide（bound method），
# 签名只有 (pre_process, post_process, vp_stage)，不收 config/pg_collection。
class FakeBridgeProvider:
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        return {"pre": pre_process, "post": post_process, "vp": vp_stage}

bound = FakeBridgeProvider().provide
wrapped = wrap(bound, args=None)

# 1) 包装后暴露的签名必须与裸 provider 一致（否则 Megatron 的 build_model 会
#    误判它「能收 config」而把 config 传进来 -> 坑 8/9）。
params = set(inspect.signature(wrapped).parameters)
expect = {"pre_process", "post_process", "vp_stage"}
if params != expect:
    failures.append(f"[bridge] 签名透传错误: 期望 {expect}, 实得 {params}")
else:
    print("[bridge] 签名透传 OK:", sorted(params))

# 2) 即便 Megatron 无条件传 config/pg_collection，包装层也要过滤掉，
#    转发给只认 (pre/post/vp) 的 provide 不能炸。
try:
    out = wrapped(pre_process=True, post_process=False, vp_stage=0,
                  config="CFG", pg_collection="PG")
    if out != {"pre": True, "post": False, "vp": 0}:
        failures.append(f"[bridge] 过滤后转发结果异常: {out}")
    else:
        print("[bridge] config/pg_collection 被正确过滤，转发 OK")
except TypeError as e:
    failures.append(f"[bridge] 未过滤多余 kwarg，撞 TypeError: {e}")

# 3) 非 bridge 模式：original_provider 是普通函数 (pre_process, post_process, vp_stage)。
def plain_provider(pre_process=True, post_process=True, vp_stage=None):
    return {"pre": pre_process, "post": post_process, "vp": vp_stage}

wrapped2 = wrap(plain_provider, args=None)
params2 = set(inspect.signature(wrapped2).parameters)
if params2 != expect:
    failures.append(f"[plain] 签名透传错误: 期望 {expect}, 实得 {params2}")
else:
    print("[plain] 签名透传 OK:", sorted(params2))

# 4) 若 provider 自带 **kwargs，则不过滤（原样转发，兼容未来真收 config 的 provider）。
def kw_provider(pre_process=True, post_process=True, **kw):
    return {"pre": pre_process, "post": post_process, "kw": kw}

wrapped3 = wrap(kw_provider, args=None)
try:
    out3 = wrapped3(pre_process=True, post_process=True, config="CFG")
    if out3.get("kw", {}).get("config") != "CFG":
        failures.append(f"[**kwargs] 不该过滤却被过滤: {out3}")
    else:
        print("[**kwargs] 含 **kwargs 时原样转发 OK")
except Exception as e:
    failures.append(f"[**kwargs] 转发异常: {e!r}")

print("=" * 60)
if failures:
    for f in failures:
        print("FAIL ->", f)
    sys.exit(1)
print("接线层自测全部通过：provider 包装层签名派发正确（坑 8/9 已修）")
'''
    proc = subprocess.run([sys.executable, "-c", script], check=False)
    raise SystemExit(proc.returncode)


# ---------------------------------------------------------------------------
# toy 数据：几十条最小样本，只为触发 rollout / train / 权重同步这条链路。
# ---------------------------------------------------------------------------
def _write_toy_jsonl(path: str, n: int = 32, hard: bool = False) -> None:
    """生成 toy MCQ 数据。hard=True 时生成有难度的数学题（用于验证 LoRA 学习）。"""
    import json
    import random as _rnd

    if not hard:
        choices = ["A", "B", "C", "D"]
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                prompt = (
                    f"Question {i}: Which letter is the {i % 4 + 1}-th option?\n"
                    "Options: A) first B) second C) third D) fourth\n"
                    "Answer within <answer> </answer>."
                )
                f.write(
                    json.dumps(
                        {"prompt": prompt, "label": choices[i % 4], "metadata": {"id": i}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"[toy] 写入 {n} 条简单样本 -> {path}")
        return

    _rnd.seed(42)
    letters = ["A", "B", "C", "D"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            kind = i % 3
            if kind == 0:
                a, b = _rnd.randint(12, 99), _rnd.randint(12, 99)
                correct = a * b
                op_str = f"{a} × {b}"
            elif kind == 1:
                a, b = _rnd.randint(100, 999), _rnd.randint(2, 9)
                correct = a + b * _rnd.randint(10, 50)
                c = _rnd.randint(10, 50)
                correct = a * b + c
                op_str = f"{a} × {b} + {c}"
            else:
                a = _rnd.randint(200, 9999)
                b = _rnd.randint(3, 13)
                correct = a % b
                op_str = f"{a} mod {b}"

            ans_idx = _rnd.randint(0, 3)
            options = []
            for j in range(4):
                if j == ans_idx:
                    options.append(correct)
                else:
                    offset = _rnd.choice([-2, -1, 1, 2, 3, -3, 10, -10])
                    options.append(correct + offset)

            opts_str = "  ".join(f"{letters[j]}) {options[j]}" for j in range(4))
            prompt = (
                f"Compute: {op_str}\n"
                f"Options: {opts_str}\n"
                "Give your final answer within <answer> </answer> tags (just the letter, e.g. <answer>A</answer>)."
            )
            label = f"<answer>{letters[ans_idx]}</answer>"
            f.write(
                json.dumps(
                    {"prompt": prompt, "label": label, "metadata": {"id": i, "correct_value": correct}},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[toy] 写入 {n} 条数学 MCQ 样本 -> {path}")


# 翻译任务（zh->en）：BLEU 是连续奖励，天然有方差，避免 MCQ 的「全对/全错」零方差。
# reference 用英文（按空白切词的标准 BLEU 最干净），模型几乎拿不到满分 -> 有学习空间。
# 用更长、更口语化/有多种合理译法的句子：单参考 BLEU 自然落到 ~0.3-0.6，
# 且高温下同一句的多次采样会显著分散 -> 组内 BLEU 方差大 -> GRPO 有学习信号。
_ZH_EN_PAIRS = [
    ("尽管那天下着倾盆大雨，他还是坚持步行去车站接她，因为他答应过绝不让她一个人等。",
     "Even though it was pouring rain that day, he insisted on walking to the station to meet her, because he had promised never to leave her waiting alone."),
    ("这家公司之所以能在激烈的竞争中存活下来，靠的不是运气，而是多年来对产品质量的执着追求。",
     "The reason this company survived the fierce competition was not luck, but its relentless pursuit of product quality over many years."),
    ("她一边收拾行李，一边回想起童年时在乡下度过的那些无忧无虑的夏天。",
     "While packing her luggage, she recalled the carefree summers she had spent in the countryside as a child."),
    ("如果我们当初多花一点时间做市场调研，也许就不会犯下这么严重的错误了。",
     "If we had spent a little more time on market research back then, we might not have made such a serious mistake."),
    ("随着人工智能技术的飞速发展，许多曾经被认为不可能的事情如今都已变成现实。",
     "With the rapid development of artificial intelligence, many things once thought impossible have now become reality."),
    ("这位老人虽然已经八十多岁，但每天清晨都会到江边打太极，几十年如一日。",
     "Although the old man is already in his eighties, he goes to the riverside every morning to practice tai chi, as he has done for decades."),
    ("会议一直开到深夜，大家都筋疲力尽，却仍然没能就预算问题达成一致意见。",
     "The meeting dragged on late into the night, and although everyone was exhausted, they still failed to reach an agreement on the budget."),
    ("他把一生的积蓄都投入到这个项目里，因此一旦失败，后果将不堪设想。",
     "He poured his entire life savings into this project, so if it fails, the consequences would be unimaginable."),
    ("孩子们刚走进博物馆，就被那座巨大的恐龙骨架深深吸引住了，久久不愿离开。",
     "As soon as the children entered the museum, they were so captivated by the huge dinosaur skeleton that they were reluctant to leave."),
    ("这本小说之所以打动了无数读者，是因为它真实地描写了普通人在困境中的挣扎与希望。",
     "This novel moved countless readers because it honestly portrayed the struggles and hopes of ordinary people in adversity."),
    ("无论你将来走到哪里，都不要忘记当初是什么让你下定决心走上这条路的。",
     "No matter where you go in the future, never forget what first made you determined to take this path."),
    ("由于交通堵塞，原本半小时的车程花了将近两个小时，他到达时早已错过了开场。",
     "Because of the traffic jam, a trip that should have taken half an hour took nearly two hours, and he had long missed the opening by the time he arrived."),
    ("政府承诺将在未来五年内大幅增加对教育和医疗的投入，以缩小城乡之间的差距。",
     "The government promised to substantially increase investment in education and healthcare over the next five years to narrow the gap between urban and rural areas."),
    ("她从不轻易向别人诉说自己的烦恼，宁愿一个人默默承受所有的压力。",
     "She never readily confides her troubles to others, preferring to bear all the pressure silently on her own."),
    ("这座古老的桥见证了这座城市几百年的兴衰，如今依然静静地横跨在河面之上。",
     "This ancient bridge has witnessed centuries of the city's rise and fall, and still spans the river quietly to this day."),
    ("尽管所有人都劝他放弃，他依然相信只要坚持下去，总有一天会看到成果。",
     "Although everyone advised him to give up, he still believed that if he kept going, one day he would see results."),
    ("那场突如其来的暴风雪让整个机场陷入瘫痪，成千上万的旅客被迫滞留在候机厅里。",
     "The sudden blizzard paralyzed the entire airport, forcing tens of thousands of passengers to be stranded in the terminal."),
    ("他说话总是不紧不慢，但每一个字都让人感觉经过了深思熟虑。",
     "He always speaks slowly and calmly, yet every word feels carefully considered."),
    ("这项研究的结果出乎所有人的意料，甚至连领导这个团队的教授都感到难以置信。",
     "The results of this study surprised everyone, and even the professor leading the team found them hard to believe."),
    ("在那个物质匮乏的年代，一家人能围坐在一起吃顿热饭，就已经是莫大的幸福了。",
     "In that era of scarcity, simply being able to sit together as a family and share a hot meal was already a great happiness."),
    ("公司高层一再强调，员工的安全永远比生产进度更重要，绝不能为了赶工而冒险。",
     "Senior management repeatedly stressed that employee safety always matters more than the production schedule, and no risks should be taken just to meet deadlines."),
    ("他望着窗外渐渐暗下来的天空，心里突然涌起一种说不清楚的失落感。",
     "Gazing at the sky gradually darkening outside the window, he was suddenly overcome by an inexplicable sense of loss."),
    ("这次旅行虽然遇到了不少麻烦，但回头想想，那些意外反而成了最难忘的回忆。",
     "Although the trip ran into quite a few troubles, looking back, those mishaps turned out to be the most unforgettable memories."),
    ("只有当你真正经历过失败，才会明白成功背后所需要付出的努力有多大。",
     "Only when you have truly experienced failure will you understand how much effort lies behind success."),
    ("她花了整整一年时间才适应这座陌生城市的生活节奏和人情冷暖。",
     "It took her an entire year to get used to the pace of life and the warmth and coldness of human relationships in this unfamiliar city."),
    ("这份报告必须在周五之前提交，否则整个项目的进度都会受到严重影响。",
     "This report must be submitted before Friday, otherwise the progress of the entire project will be seriously affected."),
    ("年轻的时候我们总以为时间用不完，等到明白珍惜的时候，许多机会早已不再。",
     "When we are young we always think there is endless time, but by the time we learn to cherish it, many opportunities are already long gone."),
    ("他凭借多年积累的经验，很快就判断出问题的根源并不在于设备本身。",
     "Drawing on years of accumulated experience, he quickly concluded that the root of the problem did not lie in the equipment itself."),
    ("无论外界如何评价，她始终保持着内心的平静，专注于自己真正热爱的事情。",
     "No matter how the outside world judged her, she always kept her inner calm and focused on what she truly loved."),
    ("这场谈判持续了好几轮，双方最终在一些关键问题上做出了妥协。",
     "The negotiation went through several rounds, and the two sides eventually made compromises on some key issues."),
    ("看到孩子终于走上了舞台，台下的母亲激动得热泪盈眶，却又强忍着不让它落下。",
     "Seeing her child finally step onto the stage, the mother in the audience was so moved that her eyes filled with tears, which she struggled to hold back."),
    ("如果不是亲眼所见，我怎么也不会相信短短几年间这个小村庄竟然发生了如此巨大的变化。",
     "Had I not seen it with my own eyes, I would never have believed that such enormous changes could take place in this small village in just a few years."),
]


def _write_translation_jsonl(path: str, n: int = 64) -> None:
    """生成 zh->en 翻译 toy 数据，label 为英文参考译文，配 bleu 奖励。"""
    import json

    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            zh, en = _ZH_EN_PAIRS[i % len(_ZH_EN_PAIRS)]
            prompt = (
                "Translate the following Chinese sentence into natural English.\n"
                f"Chinese: {zh}\n"
                "Put ONLY the English translation within <answer> </answer> tags, "
                "e.g. <answer>your translation here</answer>."
            )
            label = f"<answer>{en}</answer>"
            f.write(
                json.dumps(
                    {"prompt": prompt, "label": label, "metadata": {"id": i, "rm_type": "bleu"}},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[toy] 写入 {n} 条 zh->en 翻译样本 -> {path}")


# ChatML 兜底模板：当 checkpoint 没自带 chat_template.json 时用它把 messages 拼成串。
# 兼容 content 为字符串或多模态 part 列表两种形态。
_FALLBACK_CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}"
    "{% if part['type'] == 'text' %}{{ part['text'] }}{% endif %}"
    "{% endfor %}{% endif %}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _chat_template_kwargs() -> str:
    """返回 JSON 串供 --apply-chat-template-kwargs 用。

    Qwen3-Omni 的 chat_template 通常在 processor 的 chat_template.json 里，
    裸 AutoTokenizer 在 transformers 5.x 下加载不到。优先读 checkpoint 自带的
    chat_template.json，缺失时退回内置 ChatML 模板，确保 apply_chat_template 不崩。
    """
    import json

    ct_path = os.path.join(MODEL_DIR, "chat_template.json")
    template = None
    if os.path.exists(ct_path):
        try:
            with open(ct_path, encoding="utf-8") as f:
                template = json.load(f).get("chat_template")
            if template:
                print(f"[chat_template] 用 checkpoint 自带模板 {ct_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[chat_template] 读取 {ct_path} 失败({e})，退回内置 ChatML")
    if not template:
        template = _FALLBACK_CHATML
        print("[chat_template] 用内置 ChatML 兜底模板")
    return json.dumps({"chat_template": template})


# ---------------------------------------------------------------------------
# 主函数：在 4×A100-80GB 上起 ray + 跑冒烟脚本
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume},
    timeout=120 * 60,
)
def smoke() -> None:
    import json
    import subprocess

    smoke_script = "scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh"
    toy_path = "/root/toy.jsonl"
    _write_toy_jsonl(toy_path)

    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json，先用 modal_run.py 下载或检查 Volume。"
    )

    # 坑14: sgl-router(Rust) 注册 tokenizer 时需要 fast 格式的 tokenizer.json，
    # 但 Qwen3-Omni checkpoint 只有 vocab.json + merges.txt(slow BPE)。
    # 用 HF fast tokenizer 落一份 tokenizer.json 到模型目录（幂等）。
    tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(tok_json):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
        tok.save_pretrained(MODEL_DIR)
        assert os.path.exists(tok_json), "save_pretrained 没生成 tokenizer.json（可能不是 fast tokenizer）"
        try:
            model_volume.commit()
        except Exception:
            pass
        print(f"[坑14] 已生成 {tok_json}")
    else:
        print(f"[坑14] tokenizer.json 已存在，跳过")

    # 自己起 ray（设 RELAX_ENTRYPOINT_MODE 让冒烟脚本跳过 local.sh，
    # 避免 local.sh 里的 `pkill -9 python` 把 Modal 自身的 python 进程杀掉）。
    env = os.environ.copy()
    env.update(
        {
            "MEGATRON": "/root/Megatron-LM/",
            "RELAX": RELAX_REMOTE,
            "MODEL_CONFIG_DIR": f"{RELAX_REMOTE}/scripts/models",
            "PYTHONPATH": PYTHONPATH,
            "RELAX_ENTRYPOINT_MODE": "local",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "NUM_GPUS": "4",
            "RAY_ADDRESS": "http://127.0.0.1:8265",
            "HF_CKPT": MODEL_DIR,
            "MODEL_DIR": "/models",
            "DATA": toy_path,
            "CHAT_TEMPLATE_KWARGS": _chat_template_kwargs(),
            "MULTIMODAL_KEYS": "" if TEXT_ONLY else '{"image":"image","audio":"audio"}',
            # 本地注入的 sglang 较新，其启动期会断言 flashinfer/sgl-kernel 版本，
            # 而 slime 镜像里的是旧版（flashinfer 0.6.3）。冒烟切 triton backend 已绕开
            # flashinfer 断言，这里再跳过 sgl-kernel 版本断言（probe 已确认能 import）。
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "RUNTIME_ENV_JSON": json.dumps(
                {
                    "env_vars": {
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": PYTHONPATH,
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                    }
                }
            ),
        }
    )

    print("\n========== 起 ray head ==========")
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=env,
        check=True,
    )

    print(f"\n========== 跑冒烟脚本 {smoke_script} ==========")
    proc = subprocess.run(["bash", smoke_script], cwd=RELAX_REMOTE, env=env)
    print(f"\n[exit] smoke returncode={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


# ---------------------------------------------------------------------------
# 学习验证：用有难度的数学题验证 LoRA 确实在学（reward 上升、loss 非零）
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume, S2TT_DIR: s2tt_volume},
    timeout=240 * 60,
    # Modal GPU 函数一律可被抢占；靠卷上 ckpt + 自动重试续跑。
    retries=modal.Retries(max_retries=10, initial_delay=5.0),
)
def learn(num_rollout: int = 30, task: str = "translate") -> None:
    """跑 GRPO 训练，验证「reward 随训练上升」（效果层）。

    task:
    - "translate"（默认）：zh->en 翻译 + BLEU 连续奖励。BLEU∈(0,1) 天然有方差，
      模型几乎拿不到满分 → 有学习空间，避免 MCQ 的「全对/全错」零方差。
    - "math"：2-3 位数乘法/模运算 MCQ + 0/1 奖励（已知对 30B 太易，留作对照）。

    其他：
    - 步数：num_rollout（默认 30，足够看趋势又不太烧钱）
    - LR：5e-5；温度：1.0（采样多样性，组内有好坏混合）

    成功标准（跑后自动解析 reward 曲线判定）：
    - 学习信号存在：raw_reward 有方差（BLEU 基本恒满足）
    - 趋势上升：后 1/3 步平均 reward > 前 1/3 步平均 reward
    """
    import json
    import subprocess

    smoke_script = "scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh"
    multimodal_keys = ""
    max_prompt = ""
    # 默认（冒烟级）超参；正式实验在对应分支里覆盖。
    n_samples = "4"
    rollout_batch = "8"
    global_batch = "32"
    lr = "5e-5"
    save_dir = ""        # 设了就开启 save/load 续跑
    save_interval = "5"
    if task == "translate":
        data_path = "/root/translate.jsonl"
        _write_translation_jsonl(data_path, n=256)
        rm_type = "bleu"
        max_resp = "512"
        temperature = "1.3"  # 高温让同句多采样译文分散 -> 组内 BLEU 方差 -> 有梯度
    elif task == "math":
        data_path = "/root/math_mcq.jsonl"
        _write_toy_jsonl(data_path, n=256, hard=True)
        rm_type = "multiple_choice"
        max_resp = "1024"
        temperature = "1.0"
    elif task == "s2tt":
        # 英语音频→中文翻译（移植 my_omni）。数据由 prep_s2tt 预先下到 s2tt 卷。
        data_path = S2TT_JSONL
        assert os.path.exists(data_path), (
            f"S2TT 数据不存在 {data_path}，先跑：modal run modal_relax_smoke.py::prep"
        )
        rm_type = "bleu"
        max_resp = "512"
        temperature = "1.1"          # 略升温 -> 组内译文更分散 -> GRPO 有梯度
        multimodal_keys = '{"audio": "audios"}'
        max_prompt = "4096"          # 音频 token 多，放宽 prompt 长度上限
        n_samples = "8"              # 关键杠杆：组内更多采样 -> 更稳的相对优势
        rollout_batch = "8"
        global_batch = "64"          # = rollout_batch * n_samples
        lr = "1e-4"                  # 探路用稍大 LR，40 步内看清趋势
        save_dir = "/s2tt/ckpt/s2tt_probe"   # 存档+续跑（抗抢占）
        save_interval = "5"
    else:
        raise ValueError(f"unknown task={task!r}")

    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json"
    )

    tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(tok_json):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
        tok.save_pretrained(MODEL_DIR)
        try:
            model_volume.commit()
        except Exception:
            pass
        print(f"[learn] 已生成 {tok_json}")

    env = os.environ.copy()
    env.update(
        {
            "MEGATRON": "/root/Megatron-LM/",
            "RELAX": RELAX_REMOTE,
            "MODEL_CONFIG_DIR": f"{RELAX_REMOTE}/scripts/models",
            "PYTHONPATH": PYTHONPATH,
            "RELAX_ENTRYPOINT_MODE": "local",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "NUM_GPUS": "4",
            "RAY_ADDRESS": "http://127.0.0.1:8265",
            "HF_CKPT": MODEL_DIR,
            "MODEL_DIR": "/models",
            "DATA": data_path,
            "NUM_ROLLOUT": str(num_rollout),
            "RM_TYPE": rm_type,
            "LR": lr,
            "N_SAMPLES": n_samples,
            "ROLLOUT_BATCH": rollout_batch,
            "GLOBAL_BATCH": global_batch,
            "SAVE_DIR": save_dir,
            "SAVE_INTERVAL": save_interval,
            "ROLLOUT_TEMPERATURE": temperature,
            "ROLLOUT_MAX_RESPONSE_LEN": max_resp,
            "ROLLOUT_MAX_PROMPT_LEN": max_prompt,
            "CHAT_TEMPLATE_KWARGS": _chat_template_kwargs(),
            "MULTIMODAL_KEYS": multimodal_keys,
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "RUNTIME_ENV_JSON": json.dumps(
                {
                    "env_vars": {
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": PYTHONPATH,
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                    }
                }
            ),
        }
    )

    print("\n========== 起 ray head ==========")
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=env,
        check=True,
    )

    log_path = "/root/learn_full.log"
    print(
        f"\n========== 学习验证 (task={task}, rm={rm_type}, {num_rollout} steps, "
        f"lr={lr}, n_samples={n_samples}, gbs={global_batch}, save={save_dir or 'off'}) =========="
    )
    proc = subprocess.run(
        f"bash {smoke_script} 2>&1 | tee {log_path}",
        shell=True,
        cwd=RELAX_REMOTE,
        env=env,
    )
    print(f"\n[exit] learn returncode={proc.returncode}")
    if save_dir:
        # 持久化最终 ckpt（被抢占时靠 Modal 后台 commit + 重试续跑，这里确保成功路径落盘）。
        try:
            s2tt_volume.commit()
            print(f"[learn] 已 commit s2tt 卷（ckpt 在 {save_dir}）")
        except Exception as e:  # noqa: BLE001
            print(f"[learn] commit 卷失败（忽略）：{e}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("\n========== 解析 reward 曲线 ==========")
    curve = _parse_reward_curve(log_path)
    if not curve:
        print("[FAIL] 日志里没解析到任何 reward 行，无法判定")
        raise SystemExit(1)

    raws = [r for _, r, _ in curve]
    for step, raw, rew in curve:
        bar = "#" * int(round(raw * 40))
        print(f"  step {step:2d}: raw_reward={raw:.3f} rewards={rew:.3f} |{bar}")

    n = len(raws)
    k = max(1, n // 3)
    first = sum(raws[:k]) / k
    last = sum(raws[-k:]) / k
    rmin, rmax = min(raws), max(raws)
    # 学习信号：raw_reward 出现过 (0,1) 之间的值，或步间有差异 → 组内有方差 → 梯度非零
    has_signal = any(0.0 < r < 1.0 for r in raws) or len({round(r, 4) for r in raws}) > 1
    trend_up = last > first + 1e-6

    print("\n========== 同步耗时（perf）==========")
    perf = _parse_perf_curve(log_path)
    if not perf:
        print("  [warn] 没解析到 perf 行（可能日志被截断）")
    else:
        def _avg(key):
            vals = [d[key] for d in perf.values() if key in d]
            return sum(vals) / len(vals) if vals else None

        wake = _avg("perf/wake_up_time")
        sleep = _avg("perf/sleep_time")
        upd = _avg("perf/update_weights_time")
        train = _avg("perf/train_time")
        rollout_t = _avg("perf/rollout_time")
        step_t = _avg("perf/step_time")
        for label, v in [
            ("wake_up(resume onload)", wake),
            ("sleep(offload)", sleep),
            ("update_weights(同步+LoRA热推)", upd),
            ("train", train),
            ("rollout", rollout_t),
            ("step(总)", step_t),
        ]:
            print(f"  {label:32}: {v:.2f}s" if v is not None else f"  {label:32}: -")
        sync = sum(x for x in (wake, sleep, upd) if x is not None)
        denom = step_t or ((train or 0) + (rollout_t or 0) + sync) or None
        if denom:
            print(f"  → 同步开销(wake+sleep+update) ≈ {sync:.2f}s / 整步 {denom:.2f}s = {sync / denom:.0%}")

    print("\n========== 结论 ==========")
    print(f"  步数={n}  reward范围=[{rmin:.3f}, {rmax:.3f}]")
    print(f"  前 {k} 步均值={first:.3f}  后 {k} 步均值={last:.3f}  Δ={last - first:+.3f}")
    print(f"  学习信号(有方差/梯度非零): {'✅ 有' if has_signal else '❌ 无(reward 恒定)'}")
    print(f"  reward 趋势: {'✅ 上升' if trend_up else '⚠️ 未上升(可能步数不够/需调参)'}")
    if not has_signal:
        print("\n[FAIL] reward 无方差 → advantage 恒 0 → 没有学习信号（任务太易/太难或 reward 配置问题）")
        raise SystemExit(1)
    if trend_up:
        print("\n[PASS] 效果层验证：存在学习信号且 reward 随训练上升。")
    else:
        print("\n[PARTIAL] 有学习信号(梯度非零)，但本次步数内 reward 未见上升；"
              "机制已通，属 RL 调参/步数问题。")


def _check_rollout_coherence(pt_path: str) -> dict:
    """加载一个 rollout dump(.pt),判断输出是否连贯(非乱码)。

    坑23 症状:base 被 offload/resume 搞坏后,所有 rollout 输出变乱码
    (大量重复单一 token / 停止符刷屏 / 非 ASCII 乱码)。这里对每个样本
    的解码文本做几项启发式检查,返回汇总。
    """
    import torch

    obj = torch.load(pt_path, weights_only=False)
    samples = obj.get("samples", [])
    results = []
    for s in samples:
        decoded = s.get("decoded_tokens") or []
        # 优先用 response 长度截取生成段;拿不到就用全序列。
        resp_len = s.get("response_length") or s.get("response_len")
        toks = decoded[-int(resp_len):] if resp_len else decoded
        text = "".join(toks)
        n = len(toks)
        if n == 0:
            results.append({"ok": False, "reason": "empty", "text": ""})
            continue
        # 1) 单一 token 占比(乱码常表现为同一 token 刷屏)
        from collections import Counter
        most_common_frac = Counter(toks).most_common(1)[0][1] / n
        # 2) 可打印 ASCII 占比(乱码常含大量非常规字符)
        printable = sum(1 for c in text if 32 <= ord(c) < 127 or c in "\n\t ")
        ascii_frac = printable / max(1, len(text))
        ok = (most_common_frac < 0.5) and (ascii_frac > 0.5) and (n >= 2)
        reason = []
        if most_common_frac >= 0.5:
            reason.append(f"单token占比{most_common_frac:.0%}")
        if ascii_frac <= 0.5:
            reason.append(f"ASCII占比{ascii_frac:.0%}")
        if n < 2:
            reason.append("过短")
        results.append({
            "ok": ok,
            "reason": ",".join(reason) or "coherent",
            "text": text[:120],
            "most_common_frac": round(most_common_frac, 3),
            "ascii_frac": round(ascii_frac, 3),
            "n": n,
        })
    n_ok = sum(1 for r in results if r["ok"])
    return {"path": pt_path, "n_samples": len(results), "n_ok": n_ok, "samples": results}


def _parse_reward_curve(log_path: str) -> list:
    """从训练日志里抽每步的 reward（relax.backends.megatron.data:313 打印的那行）。

    形如：rollout 0: {'rollout/raw_reward': 0.4, ..., 'rollout/rewards': 0.4, ...}
    返回 [(step, raw_reward, rewards), ...]，按 step 排序、去重。
    """
    import re

    pat = re.compile(
        r"rollout (\d+): \{[^}]*'rollout/raw_reward': ([-\d.eE]+)"
        r"[^}]*'rollout/rewards': ([-\d.eE]+)"
    )
    seen: dict[int, tuple] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                seen[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return [(s, seen[s][0], seen[s][1]) for s in sorted(seen)]


def _parse_perf_curve(log_path: str) -> dict:
    """从训练日志抽每步的 perf 计时（train_metric_utils.py:47 的 `perf N: {...}` 行）。

    重点关心同步开销：
    - perf/wake_up_time     : resume 训练模型显存（onload）
    - perf/sleep_time       : offload 训练模型显存
    - perf/update_weights_time : 权重同步（含 SGLang onload + LoRA 热推 IPC）
    - perf/train_time / perf/step_time : 训练 / 整步耗时（做分母看占比）
    返回 {step: {key: val}}。
    """
    import re

    keys = (
        "perf/wake_up_time",
        "perf/sleep_time",
        "perf/update_weights_time",
        "perf/train_time",
        "perf/step_time",
        "perf/rollout_time",
    )
    step_pat = re.compile(r"perf (\d+): \{")
    out: dict[int, dict] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            sm = step_pat.search(line)
            if not sm:
                continue
            step = int(sm.group(1))
            d = {}
            for key in keys:
                km = re.search(rf"'{re.escape(key)}': ([-\d.eE]+)", line)
                if km:
                    d[key] = float(km.group(1))
            if d:
                out[step] = d
    return out


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume},
    timeout=120 * 60,
)
def verify_cpu_backup() -> None:
    """端到端验证坑23 修复:跑完整 colocate LoRA RL 2 步,开 --dump-details,
    跑后解码 rollout dump,判断 base 在 offload→resume→热加载 后是否仍连贯。

    成功标准:rollout 1(经历过完整 train→offload→resume→热加载 周期)的
    样本解码文本连贯(非乱码),且与 rollout 0 同等连贯。
    """
    import json
    import subprocess

    smoke_script = "scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh"
    toy_path = "/root/toy.jsonl"
    dump_dir = "/root/cpu_backup_dump"
    _write_toy_jsonl(toy_path)

    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json"
    )

    tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(tok_json):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
        tok.save_pretrained(MODEL_DIR)
        try:
            model_volume.commit()
        except Exception:
            pass
        print(f"[verify_cpu_backup] 已生成 {tok_json}")

    env = os.environ.copy()
    env.update(
        {
            "MEGATRON": "/root/Megatron-LM/",
            "RELAX": RELAX_REMOTE,
            "MODEL_CONFIG_DIR": f"{RELAX_REMOTE}/scripts/models",
            "PYTHONPATH": PYTHONPATH,
            "RELAX_ENTRYPOINT_MODE": "local",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "NUM_GPUS": "4",
            "RAY_ADDRESS": "http://127.0.0.1:8265",
            "HF_CKPT": MODEL_DIR,
            "MODEL_DIR": "/models",
            "DATA": toy_path,
            "NUM_ROLLOUT": "2",
            "DUMP_DETAILS": dump_dir,
            "CHAT_TEMPLATE_KWARGS": _chat_template_kwargs(),
            "MULTIMODAL_KEYS": "" if TEXT_ONLY else '{"image":"image","audio":"audio"}',
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "RUNTIME_ENV_JSON": json.dumps(
                {
                    "env_vars": {
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": PYTHONPATH,
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                    }
                }
            ),
        }
    )

    print("\n========== 起 ray head ==========")
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=env,
        check=True,
    )

    print(f"\n========== 跑端到端冒烟(2 步 + dump)==========")
    proc = subprocess.run(["bash", smoke_script], cwd=RELAX_REMOTE, env=env)
    print(f"\n[exit] verify_cpu_backup returncode={proc.returncode}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    print("\n========== 解码 rollout dump,判断 base 是否存活 ==========")
    rollout_dir = os.path.join(dump_dir, "rollout_data")
    if not os.path.isdir(rollout_dir):
        print(f"[FAIL] 没找到 dump 目录 {rollout_dir}")
        raise SystemExit(1)

    reports = []
    for rid in ("0", "1"):
        pt = os.path.join(rollout_dir, f"{rid}.pt")
        if not os.path.exists(pt):
            print(f"[WARN] 缺 {pt}")
            continue
        rep = _check_rollout_coherence(pt)
        reports.append((rid, rep))
        print(f"\n--- rollout {rid}: {rep['n_ok']}/{rep['n_samples']} 连贯 ---")
        for i, s in enumerate(rep["samples"][:4]):
            flag = "OK " if s["ok"] else "BAD"
            print(f"  [{flag}] ({s['reason']}) n={s.get('n')} '{s['text']}'")

    print("\n========== 结论 ==========")
    if not reports:
        print("[FAIL] 没有任何 rollout dump 可判断")
        raise SystemExit(1)
    all_ok = True
    for rid, rep in reports:
        frac = rep["n_ok"] / max(1, rep["n_samples"])
        verdict = "PASS" if frac >= 0.8 else "FAIL"
        if frac < 0.8:
            all_ok = False
        print(f"  rollout {rid}: {rep['n_ok']}/{rep['n_samples']} 连贯 -> {verdict}")
    if all_ok:
        print("\n[PASS] base 在 offload→resume→热加载 后存活,坑23 修复端到端验证通过。")
    else:
        print("\n[FAIL] 有 rollout 输出乱码,base 可能未正确恢复,需复查 enable_weights_cpu_backup。")
        raise SystemExit(1)


@app.function(image=image, volumes={"/models": model_volume}, timeout=10 * 60)
def inspect_ckpt() -> None:
    """便宜地（纯 CPU）查实 Volume 里 Qwen3-Omni checkpoint 的 tokenizer/模板文件。

    目的：搞清楚 chat_template 到底在不在、在哪个文件里，避免凭空假设。
    """
    import json

    print("=" * 60)
    print(f"[inspect] 列目录 {MODEL_DIR}")
    try:
        files = sorted(os.listdir(MODEL_DIR))
    except Exception as e:  # noqa: BLE001
        print(f"[inspect] 列目录失败: {e}")
        return
    for fn in files:
        print("   ", fn)

    # 1) 直接看几个可能放 chat_template 的文件
    for cand in ["tokenizer_config.json", "chat_template.json", "chat_template.jinja", "processor_config.json", "preprocessor_config.json"]:
        p = os.path.join(MODEL_DIR, cand)
        if not os.path.exists(p):
            print(f"[inspect] {cand}: 不存在")
            continue
        if cand.endswith(".json"):
            try:
                with open(p, encoding="utf-8") as f:
                    obj = json.load(f)
                has_ct = isinstance(obj, dict) and ("chat_template" in obj)
                print(f"[inspect] {cand}: 存在；含 chat_template? {has_ct}")
                if has_ct:
                    ct = obj["chat_template"]
                    print(f"           chat_template 长度={len(ct)}，前 80 字符: {ct[:80]!r}")
            except Exception as e:  # noqa: BLE001
                print(f"[inspect] {cand}: 读取失败 {e}")
        else:
            sz = os.path.getsize(p)
            print(f"[inspect] {cand}: 存在（{sz} bytes）")

    # 2) 用 transformers 实际加载，看 AutoTokenizer / AutoProcessor 谁拿到 chat_template
    print("-" * 60)
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"[inspect] AutoTokenizer.chat_template 是否存在: {getattr(tok, 'chat_template', None) is not None}")
    except Exception as e:  # noqa: BLE001
        print(f"[inspect] AutoTokenizer 加载失败: {e}")
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"[inspect] AutoProcessor.chat_template 是否存在: {getattr(proc, 'chat_template', None) is not None}")
    except Exception as e:  # noqa: BLE001
        print(f"[inspect] AutoProcessor 加载失败: {e}")

    # 3) 坑14: sgl-router(Rust) 需要 fast 的 tokenizer.json；Qwen3-Omni 只有 vocab.json+merges.txt。
    #    在便宜的 CPU 函数里预生成并 commit 到 Volume，A100 冒烟那次就能直接跳过。
    print("-" * 60)
    tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if os.path.exists(tok_json):
        print(f"[inspect] tokenizer.json 已存在，跳过生成")
    else:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
            tok.save_pretrained(MODEL_DIR)
            ok = os.path.exists(tok_json)
            print(f"[inspect] 生成 tokenizer.json: {ok}")
            if ok:
                model_volume.commit()
                print("[inspect] 已 commit 到 Volume")
        except Exception as e:  # noqa: BLE001
            print(f"[inspect] 生成 tokenizer.json 失败: {e}")


@app.local_entrypoint()
def inspect() -> None:
    inspect_ckpt.remote()


# ---------------------------------------------------------------------------
# Tier 1（单卡 T4，几分钟）：真实 Megatron 模块 + 真实 wrap/attach/convert 链路，
# 不加载 30B 权重。对应 verify_lora_e2e.py（Part A 必跑、Part B 真实 bridge 尽力）。
# ---------------------------------------------------------------------------
@app.function(image=image, gpu="L4", volumes={"/models": model_volume}, timeout=30 * 60)
def run_tier1() -> None:
    import subprocess
    import sys

    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    proc = subprocess.run([sys.executable, "/root/verify_lora_e2e.py"], check=False, env=env)
    raise SystemExit(proc.returncode)


@app.local_entrypoint()
def tier1() -> None:
    run_tier1.remote()


# ---------------------------------------------------------------------------
# 坑 22 诊断：SGLang 独立推理 Qwen3-Omni（不走 Relax，不加 LoRA）
# 验证 base model 在 SGLang 中能否正常生成文本。
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume},
    timeout=60 * 60,
)
def sglang_standalone() -> None:
    """启动 SGLang server，不加 LoRA / 不走 Relax，直接对 base model 发一条请求验证输出。"""
    import subprocess
    import sys

    script = r'''
import os, time, requests, json

os.environ["PYTHONPATH"] = "/root/sglang_src/python:/root/Relax:/root/Megatron-LM"
os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server

server_args = ServerArgs(
    model_path="/models/qwen3-omni",
    trust_remote_code=True,
    host="127.0.0.1",
    port=30000,
    tp_size=4,
    mem_fraction_static=0.7,
    attention_backend="triton",
    disable_cuda_graph=True,
    disable_custom_all_reduce=True,
    skip_server_warmup=True,
    enable_lora=True,
    max_lora_rank=16,
    max_loras_per_batch=1,
    lora_target_modules=["qkv_proj", "o_proj"],
)

print(f"[sglang-standalone] Starting server with tp=4, model=/models/qwen3-omni")
p = multiprocessing.Process(target=launch_server, args=(server_args,))
p.start()

# Wait for server healthy (warmup compiles MoE triton kernels, can take 3-5 min)
base_url = "http://127.0.0.1:30000"
for i in range(180):
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            print(f"[sglang-standalone] Server healthy after {i*5}s")
            break
    except Exception:
        pass
    time.sleep(5)
else:
    print("[sglang-standalone] TIMEOUT waiting for server health")
    p.kill()
    raise SystemExit(1)

# Send a simple generate request
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+3?<|im_end|>\n<|im_start|>assistant\n"
payload = {
    "text": prompt,
    "sampling_params": {
        "max_new_tokens": 64,
        "temperature": 0.0,
    },
    "return_logprob": True,
    "top_logprobs_num": 5,
}

print(f"[sglang-standalone] Sending request...")
r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
result = r.json()
print(f"[sglang-standalone] Status: {r.status_code}")
print(f"[sglang-standalone] Output text: {repr(result.get('text', ''))[:500]}")

meta = result.get("meta_info", {})
print(f"[sglang-standalone] Completion tokens: {meta.get('completion_tokens', '?')}")
if "input_token_logprobs" in result:
    logprobs = result.get("output_token_logprobs", [])
    if logprobs:
        avg_lp = sum(lp[0] for lp in logprobs[:10] if lp) / min(10, len(logprobs))
        print(f"[sglang-standalone] Avg output logprob (first 10 tokens): {avg_lp:.4f}")
        print(f"[sglang-standalone] (uniform=-11.93, should be much higher if model works)")

# Also try without chat template (raw completion)
prompt2 = "The capital of France is"
payload2 = {
    "text": prompt2,
    "sampling_params": {"max_new_tokens": 32, "temperature": 0.0},
}
r2 = requests.post(f"{base_url}/generate", json=payload2, timeout=60)
result2 = r2.json()
print(f"\n[sglang-standalone] Raw completion test:")
print(f"[sglang-standalone] Input: {repr(prompt2)}")
print(f"[sglang-standalone] Output: {repr(result2.get('text', ''))[:300]}")

# ---- Test 3: Load a zero-init LoRA and check if model still works ----
print(f"\n{'='*60}")
print("[sglang-standalone] TEST 3: Load zero-init LoRA adapter")
import torch, pickle, numpy as np, sys
try:
    import pybase64
except ImportError:
    import base64 as pybase64

# Get model info to know the dimensions
info_r = requests.get(f"{base_url}/get_model_info", timeout=10)
model_info = info_r.json()
print(f"[sglang-standalone] Model info keys: {list(model_info.keys())}")

# Build zero-init LoRA weights matching the smoke script config:
# rank=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
# hidden_size=2048, num_heads=32, num_kv_heads=4, head_dim=128
# q_proj: [2048, 4096], k_proj: [2048, 512], v_proj: [2048, 512], o_proj: [4096, 2048]
rank = 16
hidden_size = 2048
head_dim = 128
num_heads = 32
num_kv_heads = 4
q_dim = num_heads * head_dim  # 4096
kv_dim = num_kv_heads * head_dim  # 512
num_layers = 48

named_tensors = []
for i in range(num_layers):
    prefix = f"thinker.model.layers.{i}.self_attn"
    # q_proj: lora_A [rank, hidden_size], lora_B [q_dim, rank]
    named_tensors.append((f"{prefix}.q_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors.append((f"{prefix}.q_proj.lora_B.weight", torch.zeros(q_dim, rank)))
    # k_proj
    named_tensors.append((f"{prefix}.k_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors.append((f"{prefix}.k_proj.lora_B.weight", torch.zeros(kv_dim, rank)))
    # v_proj
    named_tensors.append((f"{prefix}.v_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors.append((f"{prefix}.v_proj.lora_B.weight", torch.zeros(kv_dim, rank)))
    # o_proj: lora_A [rank, q_dim], lora_B [hidden_size, rank]
    named_tensors.append((f"{prefix}.o_proj.lora_A.weight", torch.randn(rank, q_dim) * 0.01))
    named_tensors.append((f"{prefix}.o_proj.lora_B.weight", torch.zeros(hidden_size, rank)))

print(f"[sglang-standalone] Built {len(named_tensors)} LoRA tensors")

# Pack into FlattenedTensorBucket format (same as Relax does)
sys.path.insert(0, "/root/Relax")
from relax.backends.megatron.sglang import FlattenedTensorBucket
bucket = FlattenedTensorBucket(named_tensors=named_tensors)
flattened_data = {
    "flattened_tensor": bucket.get_flattened_tensor().cpu(),
    "metadata": bucket.get_metadata(),
}
serialized = pybase64.b64encode(pickle.dumps(flattened_data)).decode("utf-8")

config_dict = {
    "peft_type": "LORA",
    "r": rank,
    "lora_alpha": 32.0,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
}

# Load the LoRA adapter
lora_payload = {
    "lora_name": "test_zero",
    "serialized_tensors": serialized,
    "config_dict": config_dict,
    "load_format": "flattened_bucket",
}
lr = requests.post(f"{base_url}/load_lora_adapter_from_tensors", json=lora_payload, timeout=120)
print(f"[sglang-standalone] Load LoRA status: {lr.status_code}, response: {lr.text[:200]}")

# Generate WITH the zero-init LoRA
payload3 = {
    "text": prompt,
    "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
    "lora_path": "test_zero",
    "return_logprob": True,
    "top_logprobs_num": 5,
}
r3 = requests.post(f"{base_url}/generate", json=payload3, timeout=120)
result3 = r3.json()
print(f"\n[sglang-standalone] WITH LoRA (zero-init B) test:")
print(f"[sglang-standalone] Status: {r3.status_code}")
print(f"[sglang-standalone] Output text: {repr(result3.get('text', ''))[:500]}")
meta3 = result3.get("meta_info", {})
print(f"[sglang-standalone] Completion tokens: {meta3.get('completion_tokens', '?')}")
logprobs3 = result3.get("output_token_logprobs", [])
if logprobs3:
    avg_lp3 = sum(lp[0] for lp in logprobs3[:10] if lp) / min(10, len(logprobs3))
    print(f"[sglang-standalone] Avg output logprob (first 10): {avg_lp3:.4f}")
    print(f"[sglang-standalone] (should be similar to base model, NOT -11.93)")

# ---- Test: Load LoRA from disk (PEFT adapter directory) ----
print(f"\n{'='*60}")
print("[sglang-standalone] TEST: Load LoRA from PEFT adapter dir (磁盘加载)")
import tempfile, safetensors.torch

adapter_dir = "/tmp/test_peft_adapter"
os.makedirs(adapter_dir, exist_ok=True)

# Save as safetensors (standard PEFT format)
state_dict = {name: t for name, t in named_tensors}
safetensors.torch.save_file(state_dict, f"{adapter_dir}/adapter_model.safetensors")

# Write adapter_config.json (PEFT format)
adapter_config = {
    "peft_type": "LORA",
    "base_model_name_or_path": "/models/qwen3-omni",
    "r": rank,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
with open(f"{adapter_dir}/adapter_config.json", "w") as f:
    json.dump(adapter_config, f)

print(f"[sglang-standalone] Saved PEFT adapter to {adapter_dir}")

# Load via the from-disk path
lr_disk = requests.post(
    f"{base_url}/load_lora_adapter",
    json={"lora_name": "test_disk", "lora_path": adapter_dir},
    timeout=120,
)
print(f"[sglang-standalone] Load from disk status: {lr_disk.status_code}, response: {lr_disk.text[:200]}")

# Generate with disk-loaded adapter
payload_disk = {
    "text": prompt,
    "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
    "lora_path": "test_disk",
    "return_logprob": True,
    "top_logprobs_num": 5,
}
r_disk = requests.post(f"{base_url}/generate", json=payload_disk, timeout=120)
result_disk = r_disk.json()
print(f"\n[sglang-standalone] WITH LoRA from disk (zero-init B) test:")
print(f"[sglang-standalone] Status: {r_disk.status_code}")
print(f"[sglang-standalone] Output text: {repr(result_disk.get('text', ''))[:500]}")
meta_disk = result_disk.get("meta_info", {})
print(f"[sglang-standalone] Completion tokens: {meta_disk.get('completion_tokens', '?')}")
logprobs_disk = result_disk.get("output_token_logprobs", [])
if logprobs_disk:
    avg_lp_disk = sum(lp[0] for lp in logprobs_disk[:10] if lp) / min(10, len(logprobs_disk))
    print(f"[sglang-standalone] Avg output logprob (first 10): {avg_lp_disk:.4f}")
    print(f"[sglang-standalone] (should be similar to base model, NOT -11.93)")

# ---- Test: LoRA merge into base and generate ----
print(f"\n{'='*60}")
print("[sglang-standalone] TEST: LoRA merge (apply adapter to base weights)")
# SGLang supports merging via the /update_weights_from_tensor endpoint won't work easily.
# Instead, we test "merge" semantics: load adapter with non-zero B, generate, verify output differs from base.
# Build a LoRA with small random B (will change output slightly)
named_tensors_nz = []
for i in range(num_layers):
    prefix = f"thinker.model.layers.{i}.self_attn"
    named_tensors_nz.append((f"{prefix}.q_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors_nz.append((f"{prefix}.q_proj.lora_B.weight", torch.randn(q_dim, rank) * 0.05))
    named_tensors_nz.append((f"{prefix}.k_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors_nz.append((f"{prefix}.k_proj.lora_B.weight", torch.randn(kv_dim, rank) * 0.05))
    named_tensors_nz.append((f"{prefix}.v_proj.lora_A.weight", torch.randn(rank, hidden_size) * 0.01))
    named_tensors_nz.append((f"{prefix}.v_proj.lora_B.weight", torch.randn(kv_dim, rank) * 0.05))
    named_tensors_nz.append((f"{prefix}.o_proj.lora_A.weight", torch.randn(rank, q_dim) * 0.01))
    named_tensors_nz.append((f"{prefix}.o_proj.lora_B.weight", torch.randn(hidden_size, rank) * 0.05))

# Save as PEFT directory and load
adapter_dir_nz = "/tmp/test_peft_nonzero"
os.makedirs(adapter_dir_nz, exist_ok=True)
state_dict_nz = {name: t for name, t in named_tensors_nz}
safetensors.torch.save_file(state_dict_nz, f"{adapter_dir_nz}/adapter_model.safetensors")
with open(f"{adapter_dir_nz}/adapter_config.json", "w") as f:
    json.dump(adapter_config, f)

lr_nz = requests.post(
    f"{base_url}/load_lora_adapter",
    json={"lora_name": "test_nonzero", "lora_path": adapter_dir_nz},
    timeout=120,
)
print(f"[sglang-standalone] Load non-zero LoRA status: {lr_nz.status_code}")

payload_nz = {
    "text": prompt,
    "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
    "lora_path": "test_nonzero",
    "return_logprob": True,
    "top_logprobs_num": 5,
}
r_nz = requests.post(f"{base_url}/generate", json=payload_nz, timeout=120)
result_nz = r_nz.json()
text_nz = result_nz.get('text', '')
print(f"[sglang-standalone] Non-zero LoRA output: {repr(text_nz)[:500]}")
logprobs_nz = result_nz.get("output_token_logprobs", [])
avg_lp_nz = -99.0
if logprobs_nz:
    avg_lp_nz = sum(lp[0] for lp in logprobs_nz[:10] if lp) / min(10, len(logprobs_nz))
    print(f"[sglang-standalone] Avg logprob (non-zero LoRA): {avg_lp_nz:.4f}")

# ---- Automated assertions ----
print(f"\n{'='*60}")
print("[sglang-standalone] AUTOMATED CHECKS:")
sys.stdout.flush()

base_text = result.get('text', '')
text_hot = result3.get('text', '')
text_d = result_disk.get('text', '')

checks_passed = 0
checks_total = 0

# Check 1: base model generates coherent text (not garbage)
checks_total += 1
if len(base_text.strip()) > 3 and not all(ord(c) > 0x4000 for c in base_text.strip()[:20]):
    print(f"  [PASS] Base generation coherent: {repr(base_text.strip()[:80])}")
    checks_passed += 1
else:
    print(f"  [FAIL] Base generation looks like garbage: {repr(base_text)[:100]}")

# Check 2: zero-init LoRA hot-load output is coherent
checks_total += 1
if len(text_hot.strip()) > 3:
    print(f"  [PASS] Hot-loaded LoRA (zero B) generation coherent: {repr(text_hot.strip()[:80])}")
    checks_passed += 1
else:
    print(f"  [FAIL] Hot-loaded LoRA generation broken: {repr(text_hot)[:100]}")

# Check 3: disk-loaded LoRA output is coherent
checks_total += 1
if len(text_d.strip()) > 3:
    print(f"  [PASS] Disk-loaded LoRA generation coherent: {repr(text_d.strip()[:80])}")
    checks_passed += 1
else:
    print(f"  [FAIL] Disk-loaded LoRA generation broken: {repr(text_d)[:100]}")

# Check 4: non-zero LoRA output is coherent AND differs from base
checks_total += 1
if len(text_nz.strip()) > 3:
    differs = (text_nz.strip() != base_text.strip())
    if differs:
        print(f"  [PASS] Non-zero LoRA output differs from base (LoRA is effective)")
        print(f"         Base:  {repr(base_text.strip()[:60])}")
        print(f"         LoRA:  {repr(text_nz.strip()[:60])}")
    else:
        print(f"  [WARN] Non-zero LoRA output same as base (LoRA scale too small?)")
        print(f"         Output: {repr(text_nz.strip()[:60])}")
    checks_passed += 1
else:
    print(f"  [FAIL] Non-zero LoRA generation broken: {repr(text_nz)[:100]}")

print(f"\n  RESULT: {checks_passed}/{checks_total} checks passed")
sys.stdout.flush()

if checks_passed < checks_total:
    print("[sglang-standalone] SOME CHECKS FAILED")
    p.kill()
    raise SystemExit(1)

p.kill()
print("\n[sglang-standalone] ALL TESTS PASSED")
'''
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run([sys.executable, "-u", "-c", script], check=False, env=env)
    raise SystemExit(proc.returncode)


@app.local_entrypoint()
def standalone() -> None:
    sglang_standalone.remote()


# ---------------------------------------------------------------------------
# 坑 22 分层诊断：在完整 Relax colocate 环境下，用简单/困难数据各跑 2 step，
# 并注入检查点打印 LoRA load 结果、rollout 首 token log prob 等关键诊断信息。
# 一次运行区分：数据格式问题 vs LoRA 加载失败 vs colocate memory 问题。
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume},
    timeout=120 * 60,
)
def diagnose() -> None:
    """分层诊断 rollout 输出均匀分布的根因。

    Phase 1: 简单数据 + 2 step（与 smoke 完全相同）→ 验证 colocate LoRA 基线
    Phase 2: 困难数据 + 2 step + max_response_len=1024 → 隔离数据/prompt 因素

    如果 Phase 1 也失败 → colocate/memory_saver 问题
    如果 Phase 1 成功、Phase 2 失败 → 数据格式/prompt 问题
    如果两者都成功 → 问题出在多步训练（grad buffer offload）
    """
    import json
    import subprocess

    smoke_script = "scripts/training/multimodal/run-qwen3-30B-A3B-omni-lora-smoke.sh"

    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), (
        f"Volume 里没找到模型权重 {MODEL_DIR}/config.json"
    )

    tok_json = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.exists(tok_json):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=True)
        tok.save_pretrained(MODEL_DIR)
        try:
            model_volume.commit()
        except Exception:
            pass

    base_env = os.environ.copy()
    base_env.update(
        {
            "MEGATRON": "/root/Megatron-LM/",
            "RELAX": RELAX_REMOTE,
            "MODEL_CONFIG_DIR": f"{RELAX_REMOTE}/scripts/models",
            "PYTHONPATH": PYTHONPATH,
            "RELAX_ENTRYPOINT_MODE": "local",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONUNBUFFERED": "1",
            "NUM_GPUS": "4",
            "RAY_ADDRESS": "http://127.0.0.1:8265",
            "HF_CKPT": MODEL_DIR,
            "MODEL_DIR": "/models",
            "CHAT_TEMPLATE_KWARGS": _chat_template_kwargs(),
            "MULTIMODAL_KEYS": "",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
            "RUNTIME_ENV_JSON": json.dumps(
                {
                    "env_vars": {
                        "PYTHONUNBUFFERED": "1",
                        "PYTHONPATH": PYTHONPATH,
                        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
                        "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                    }
                }
            ),
        }
    )

    print("\n========== 起 ray head ==========")
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=base_env,
        check=True,
    )

    # ============================================================
    # Phase 1: 简单数据 2 step（与 smoke 完全相同）
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: 简单数据 (1+1=?) + 2 step — 验证 colocate LoRA 基线")
    print("=" * 70)
    easy_data = "/root/diag_easy.jsonl"
    _write_toy_jsonl(easy_data, n=64, hard=False)

    env1 = {**base_env, "DATA": easy_data, "NUM_ROLLOUT": "2"}
    proc1 = subprocess.run(["bash", smoke_script], cwd=RELAX_REMOTE, env=env1)
    phase1_ok = proc1.returncode == 0
    print(f"\n>>> PHASE 1 结果: {'✅ 成功' if phase1_ok else '❌ 失败'} (returncode={proc1.returncode})")

    if not phase1_ok:
        print("\n>>> 诊断结论: Phase 1 失败 → colocate/memory_saver 环境问题，与数据无关。")
        print(">>> 建议: 对比 standalone LoRA 测试 / 移植 Miles 的 grad buffer 补丁。")
        raise SystemExit(1)

    # stop ray to clean up for phase 2
    subprocess.run("ray stop --force", shell=True, env=base_env)
    import time
    time.sleep(5)
    subprocess.run(
        "ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 "
        "--disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265",
        shell=True,
        env=base_env,
        check=True,
    )

    # ============================================================
    # Phase 2: 困难数据 + max_response_len=1024 + 2 step
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: 困难数据 (数学MCQ) + max_len=1024 + 2 step")
    print("=" * 70)
    hard_data = "/root/diag_hard.jsonl"
    _write_toy_jsonl(hard_data, n=64, hard=True)

    env2 = {
        **base_env,
        "DATA": hard_data,
        "NUM_ROLLOUT": "2",
        "ROLLOUT_MAX_RESPONSE_LEN": "1024",
        "ROLLOUT_TEMPERATURE": "1.0",
    }
    proc2 = subprocess.run(["bash", smoke_script], cwd=RELAX_REMOTE, env=env2)
    phase2_ok = proc2.returncode == 0
    print(f"\n>>> PHASE 2 结果: {'✅ 成功' if phase2_ok else '❌ 失败'} (returncode={proc2.returncode})")

    if phase1_ok and not phase2_ok:
        print("\n>>> 诊断结论: Phase 1 成功 + Phase 2 失败 → 困难数据 / max_len=1024 引入问题。")
        print(">>> 可能原因: prompt 格式、截断、或长序列生成触发 bug。")
    elif phase1_ok and phase2_ok:
        print("\n>>> 诊断结论: 两阶段都成功 → 问题出在多步训练 (>2 step) 后的 weight sync。")
        print(">>> 建议: 移植 Miles 的 patch_param_grad_buffer 补丁，或检查多轮 unload/load cycle。")
    print("\n>>> 诊断完成。")


@app.function(
    image=image,
    gpu="A100-80GB:4",
    volumes={"/models": model_volume},
    timeout=40 * 60,
)
def calibrate_difficulty() -> None:
    """便宜的纯推理探针：标定数学 MCQ 在不同配置下的准确率，找 ~40-70% 的甜点。

    在同一个 SGLang session 里扫多个配置（thinking 开/关 × 难度档），直接报准确率，
    避免反复烧完整 RL。命中 ~40-70% 的配置再拿去 learn() 跑 RL。
    """
    import subprocess
    import sys

    script = r'''
import os, time, re, random, requests

os.environ["PYTHONPATH"] = "/root/sglang_src/python:/root/Relax:/root/Megatron-LM"
os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server

server_args = ServerArgs(
    model_path="/models/qwen3-omni",
    trust_remote_code=True,
    host="127.0.0.1",
    port=30000,
    tp_size=4,
    mem_fraction_static=0.7,
    attention_backend="triton",
    disable_cuda_graph=True,
    disable_custom_all_reduce=True,
    skip_server_warmup=True,
)
print("[calib] starting SGLang tp=4 ...", flush=True)
p = multiprocessing.Process(target=launch_server, args=(server_args,))
p.start()

base = "http://127.0.0.1:30000"
for i in range(240):
    try:
        if requests.get(f"{base}/health", timeout=5).status_code == 200:
            print(f"[calib] healthy after {i*5}s", flush=True)
            break
    except Exception:
        pass
    time.sleep(5)
else:
    print("[calib] TIMEOUT"); p.kill(); raise SystemExit(1)

def build_prompt(q, thinking):
    # 手工拼 Qwen 对话格式（base /generate 不会自动套 chat template）。
    # 关 thinking == chat template 里 enable_thinking=False 的行为：assistant 头后注入空 think 块。
    p = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
         f"<|im_start|>user\n{q}<|im_end|>\n"
         "<|im_start|>assistant\n")
    if not thinking:
        p += "<think>\n\n</think>\n\n"
    return p

ANS = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
def extract(t):
    m = ANS.search(t)
    return m.group(1).strip() if m else t.strip()

def make_q(level, rng):
    if level == "mul2":      a, b = rng.randint(12, 99), rng.randint(12, 99); c = a*b; s = f"{a} x {b}"
    elif level == "mul3x2":  a, b = rng.randint(100, 999), rng.randint(12, 99); c = a*b; s = f"{a} x {b}"
    elif level == "mul3":    a, b = rng.randint(100, 999), rng.randint(100, 999); c = a*b; s = f"{a} x {b}"
    elif level == "mul4x3":  a, b = rng.randint(1000, 9999), rng.randint(100, 999); c = a*b; s = f"{a} x {b}"
    else:                    a, b = rng.randint(12, 99), rng.randint(12, 99); c = a*b; s = f"{a} x {b}"
    idx = rng.randint(0, 3); used = {c}; opts = []
    for j in range(4):
        if j == idx:
            opts.append(c)
        else:
            while True:
                v = c + rng.choice([-3, -2, -1, 1, 2, 3, 10, -10, 100, -100])
                if v > 0 and v not in used:
                    used.add(v); opts.append(v); break
    letters = "ABCD"
    q = (f"Compute: {s}\n"
         f"Options: A) {opts[0]}  B) {opts[1]}  C) {opts[2]}  D) {opts[3]}\n"
         "Give your final answer within <answer> </answer> tags (just the letter, e.g. <answer>A</answer>).")
    return q, letters[idx]

def eval_cfg(level, thinking, n=32, max_new=512, temp=1.0):
    rng = random.Random(0)
    prompts, labels = [], []
    for _ in range(n):
        q, letter = make_q(level, rng)
        prompts.append(build_prompt(q, thinking)); labels.append(letter)
    payload = {"text": prompts,
               "sampling_params": {"max_new_tokens": max_new, "temperature": temp}}
    outs = requests.post(f"{base}/generate", json=payload, timeout=900).json()
    correct = sum(1 for o, lab in zip(outs, labels) if extract(o["text"]) == lab)
    avg_len = sum(o.get("meta_info", {}).get("completion_tokens", 0) for o in outs) / len(outs)
    return correct / len(labels), avg_len

CONFIGS = [
    # 第二轮：保留 thinking（模型能算），靠「加大数字」让 CoT 偶尔失手 → 真方差
    ("mul3x2", True,  768),   # 3位 x 2位
    ("mul3",   True,  1024),  # 3位 x 3位
    ("mul4x3", True,  1024),  # 4位 x 3位（最难）
    # 备选杠杆：thinking 开但限长，让一部分 CoT 被截断（长度型方差）
    ("mul2",   True,  128),
    ("mul3",   True,  192),
]
print("\n========== 准确率标定 ==========", flush=True)
print(f"{'level':8} {'think':6} {'max_new':7} {'acc':>6} {'avg_len':>8}", flush=True)
results = []
for level, thinking, mx in CONFIGS:
    acc, alen = eval_cfg(level, thinking, n=32, max_new=mx, temp=1.0)
    results.append((level, thinking, mx, acc, alen))
    sweet = " <== 甜点(40-70%)" if 0.40 <= acc <= 0.70 else ""
    print(f"{level:8} {str(thinking):6} {mx:7} {acc:6.0%} {alen:8.1f}{sweet}", flush=True)

print("\n========== 建议 ==========", flush=True)
sweets = [r for r in results if 0.40 <= r[3] <= 0.70]
if sweets:
    for level, thinking, mx, acc, alen in sweets:
        print(f"  用 level={level}, enable_thinking={thinking}, max_response_len~{mx} (acc={acc:.0%})", flush=True)
else:
    near = min(results, key=lambda r: abs(r[3] - 0.55))
    print(f"  无配置正落在 40-70%；最接近的是 level={near[0]}, thinking={near[1]} (acc={near[3]:.0%})", flush=True)
    print("  可微调 max_new / 难度档再标定一次。", flush=True)
p.kill()
'''
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run([sys.executable, "-u", "-c", script], check=False, env=env)
    raise SystemExit(proc.returncode)


@app.local_entrypoint()
def calibrate() -> None:
    calibrate_difficulty.remote()


@app.local_entrypoint()
def diag() -> None:
    fc = diagnose.spawn()
    print(f"诊断已 spawn，function call ID: {fc.object_id}")
    print("去 Modal dashboard 看日志，不需要保持本地连接。")
    print("用 `modal app logs` 查看输出。")


@app.local_entrypoint()
def wiring() -> None:
    verify_wiring.remote()


@app.local_entrypoint()
def probe() -> None:
    run_probe.remote()


@app.local_entrypoint()
def cpu_backup() -> None:
    verify_cpu_backup.remote()


@app.local_entrypoint()
def learning() -> None:
    verify_learning.remote()


@app.local_entrypoint()
def learn_effect(steps: int = 30, task: str = "translate") -> None:
    learn.remote(num_rollout=steps, task=task)


@app.local_entrypoint()
def learn_audio(steps: int = 40) -> None:
    """正式音频 RL 实验：spawn 脱离本地连接，被抢占自动重试续跑（卷上 ckpt）。"""
    fc = learn.spawn(num_rollout=steps, task="s2tt")
    print(f"音频 RL 已 spawn（{steps} 步），function call ID: {fc.object_id}")
    print("脱离本地连接也会跑；被 Modal 抢占会自动在新容器从 /s2tt/ckpt/s2tt_probe 续跑。")
    print("看日志：modal app logs relax-omni-lora-smoke   （或 dashboard）")


# ---------------------------------------------------------------------------
# 音频消融测试：验证模型真在读音频（正确配对 vs 换配音频）
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100-80GB:2",          # 推理只需 TP=2，比训练省一半费用
    volumes={"/models": model_volume, S2TT_DIR: s2tt_volume},
    timeout=40 * 60,
)
def verify_audio_ablation(n_pairs: int = 2) -> None:
    """音频消融验证：「正确音频」vs「换配音频」，证明模型真在读音频内容。

    取 2*n_pairs 条 S2TT 样本组成 n_pairs 对，两两交叉：
      correct:  (audio_A→prompt_A),  (audio_B→prompt_B)
      swapped:  (audio_B→prompt_A),  (audio_A→prompt_B)

    预期：
      - correct 输出与 ref 更接近（BLEU 更高）
      - swapped 输出与 ref 差距大，且和 correct 输出文本明显不同
      → 证明模型真在听音频，而非只靠语言先验猜答案
    """
    import base64
    import json
    import multiprocessing
    import time

    import requests

    multiprocessing.set_start_method("spawn", force=True)

    # ---- 1. 从 s2tt 卷读样本 ----
    samples = []
    with open(S2TT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio_path = (rec.get("audios") or [None])[0]
            if audio_path and os.path.isfile(audio_path):
                samples.append(rec)
            if len(samples) >= n_pairs * 2:
                break
    assert len(samples) >= n_pairs * 2, (
        f"s2tt 卷里有效样本不足 {n_pairs*2} 条，先跑 prep"
    )
    print(f"[ablation] 读到 {len(samples)} 条样本，将测 {n_pairs} 对")

    # ---- 2. 构建 prompt 文本（不用 apply_chat_template，根 tokenizer 没有 chat_template）----
    # 直接用已知的 Qwen3-Omni ChatML 格式，与 smoke 日志里观察到的一致。
    AUDIO_SENTINEL = "<|audio_start|><|audio_pad|><|audio_end|>"
    PROMPT_TPL = (
        "<audio>\nPlease translate the English speech into Chinese. "
        "Only output the Chinese translation."
    )

    def build_prompt_text(prompt_raw: str) -> str:
        text = prompt_raw.replace("<audio>", AUDIO_SENTINEL)
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode_audio(path: str) -> str:
        with open(path, "rb") as f:
            raw = f.read()
        return "data:audio/wav;base64," + base64.b64encode(raw).decode()

    # ---- 3. 起 SGLang standalone server ----
    os.environ["SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"] = "1"
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.entrypoints.http_server import launch_server

    server_args = ServerArgs(
        model_path=MODEL_DIR,
        trust_remote_code=True,
        host="127.0.0.1",
        port=30100,
        tp_size=2,              # 2×A100-80GB 足够放 30B 权重（~30GB/卡）
        mem_fraction_static=0.72,
        attention_backend="triton",   # 避免 flashinfer 版本检查失败
        disable_cuda_graph=True,
        disable_custom_all_reduce=True,
        skip_server_warmup=True,
    )
    print("[ablation] 启动 SGLang server (TP=2) ...")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()
    base_url = "http://127.0.0.1:30100"
    for i in range(200):
        try:
            if requests.get(f"{base_url}/health", timeout=5).status_code == 200:
                print(f"[ablation] server healthy after {i*5}s")
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        p.kill()
        raise SystemExit("[ablation] TIMEOUT waiting for server")

    # ---- 4. 推理函数 ----
    def infer(prompt_raw: str, audio_path: str, label: str = "") -> dict:
        prompt_text = build_prompt_text(prompt_raw)
        audio_data = encode_audio(audio_path)
        payload = {
            "text": prompt_text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 128},
            "audio_data": [audio_data],
        }
        try:
            r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
            r.raise_for_status()
            out = r.json()
            text = (out.get("text") or "").strip()
        except Exception as e:
            text = f"[ERROR: {e}]"
        return {"pred": text, "ref": label}

    # ---- 5. 跑消融实验 ----
    try:
        import sacrebleu
        def bleu(hyp, ref):
            try:
                return round(sacrebleu.sentence_bleu(hyp, [ref], tokenize="zh").score / 100, 3)
            except Exception:
                return None
    except ImportError:
        def bleu(hyp, ref): return None  # noqa: E731

    print("\n" + "="*60)
    print("[ablation] 开始消融测试")
    print("="*60)
    all_pass = True
    for pair_idx in range(n_pairs):
        recA = samples[pair_idx * 2]
        recB = samples[pair_idx * 2 + 1]
        audioA = recA["audios"][0]
        audioB = recB["audios"][0]
        promptA = recA.get("prompt", PROMPT_TPL)
        promptB = recB.get("prompt", PROMPT_TPL)
        refA = (recA.get("label") or {}).get("ground_truth", "")
        refB = (recB.get("label") or {}).get("ground_truth", "")

        print(f"\n--- pair {pair_idx} ---")
        print(f"  A audio: {os.path.basename(audioA)}")
        print(f"  B audio: {os.path.basename(audioB)}")

        # correct 配对
        cA = infer(promptA, audioA, refA)
        cB = infer(promptB, audioB, refB)
        # swapped 配对（换了音频）
        sA = infer(promptA, audioB, refA)   # A的prompt + B的音频
        sB = infer(promptB, audioA, refB)   # B的prompt + A的音频

        for tag, res, ref_str in [
            ("correct-A", cA, refA), ("correct-B", cB, refB),
            ("swapped-A(B's audio)", sA, refA), ("swapped-B(A's audio)", sB, refB),
        ]:
            b = bleu(res["pred"], ref_str) if ref_str else None
            bleu_str = f"  BLEU={b}" if b is not None else ""
            print(f"  [{tag}] pred: {res['pred'][:120]}{bleu_str}")

        # 关键判断：correct vs swapped 的输出应不同
        changed_A = cA["pred"].strip() != sA["pred"].strip()
        changed_B = cB["pred"].strip() != sB["pred"].strip()
        print(f"  [check] A: 换音频后输出改变={'✅' if changed_A else '❌ 未变（音频可能被忽略）'}")
        print(f"  [check] B: 换音频后输出改变={'✅' if changed_B else '❌ 未变（音频可能被忽略）'}")
        if not changed_A or not changed_B:
            all_pass = False

    print("\n" + "="*60)
    if all_pass:
        print("[PASS] 所有对照组：换配音频后输出均改变 → 模型真在读音频内容 ✅")
    else:
        print("[WARN] 部分对照组换音频后输出未改变，需进一步排查音频注入链路 ⚠️")
    print("="*60)

    p.kill()


@app.local_entrypoint()
def audio_ablation(n_pairs: int = 2) -> None:
    verify_audio_ablation.remote(n_pairs=n_pairs)



@app.local_entrypoint()
def main() -> None:
    smoke.remote()
