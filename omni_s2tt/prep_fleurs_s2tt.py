"""生成 S2TT 训练数据：FLEURS 英语音频 → 中文参考译文。

这是 v1 `modal_relax_smoke.py::prep_s2tt` 的脱 Modal 版本，逻辑逐行照搬，
只把"写进 Modal Volume"换成"写进本地目录"。40 步那次实验用的就是它产出的
128 条 validation 数据，所以想复现曲线就别改 split 和 prompt。

FLEURS 是 N-way 平行语料：en_us 提供音频，cmn_hans_cn 按同一个 id 提供中文
参考，两边 join 起来才是一条 S2TT 样本。

依赖（datasets 必须 <3，否则 google/fleurs 这种脚本式数据集会被拒绝加载）：

    pip install "datasets>=2.19,<3" "numpy<2" soundfile librosa "huggingface_hub<0.26"

用法：

    python3 omni_s2tt/prep_fleurs_s2tt.py --out-dir /data/s2tt --limit 128

产出：

    /data/s2tt/train_s2tt.jsonl
    /data/s2tt/audio/fleurs_XXXXXXXX_en.wav
"""

from __future__ import annotations

import argparse
import json
import os
import wave


PROMPT = "<audio>\nPlease translate the English speech into Chinese. Only output the Chinese translation."


def _write_wav(path: str, arr, sr: int) -> None:
    import numpy as np

    a = (np.asarray(arr, dtype="float32").clip(-1, 1) * 32767).astype("int16")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(a.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="jsonl 与 wav 的落地目录")
    ap.add_argument("--limit", type=int, default=128, help="样本条数，0 表示不限")
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    from datasets import load_dataset

    audio_dir = os.path.join(args.out_dir, "audio")
    cache_dir = os.path.join(args.out_dir, ".hf-cache")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "train_s2tt.jsonl")

    print(f"[prep] 载入中文参考 cmn_hans_cn[{args.split}] ...", flush=True)
    tgt_by_id: dict = {}
    ds_tgt = load_dataset(
        "google/fleurs", "cmn_hans_cn", split=args.split, cache_dir=cache_dir, trust_remote_code=True
    )
    for x in ds_tgt.select_columns(["id", "transcription"]):
        t = (x.get("transcription") or "").strip()
        if t:
            tgt_by_id.setdefault(x["id"], t)
    print(f"[prep] 中文参考 {len(tgt_by_id)} 条", flush=True)

    print(f"[prep] 载入英语音频 en_us[{args.split}] 并按 id 对齐 ...", flush=True)
    ds_src = load_dataset(
        "google/fleurs", "en_us", split=args.split, cache_dir=cache_dir, trust_remote_code=True
    )

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for row in ds_src:
            cid = row["id"]
            ref = tgt_by_id.get(cid)
            if not ref:
                continue
            audio = row["audio"]
            wav_path = os.path.join(audio_dir, f"fleurs_{cid:08d}_en.wav")
            try:
                _write_wav(wav_path, audio["array"], audio.get("sampling_rate", 16000))
            except Exception as e:  # noqa: BLE001
                print(f"[prep] 跳过 id={cid}: {e}", flush=True)
                continue
            rec = {
                "prompt": PROMPT,
                "audios": [wav_path],
                "label": {"ground_truth": ref},
                "metadata": {
                    "src_lang": "en",
                    "tgt_lang": "zh",
                    "rm_type": "bleu",
                    "src_text": (row.get("transcription") or "").strip(),
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"[prep] 已写 {n} 条 ...", flush=True)
            if args.limit and n >= args.limit:
                break

    print(f"[prep] 完成：{n} 条 -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
