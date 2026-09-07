"""从 rollout_result 里算 BLEU 曲线。

Relax 每步都会把这一批样本写成一个 JSONL（`{--save}/rollout_result/train/<step>.jsonl`，
字段含 prompt / response / reward / label），这是 always-on 的，不用额外开关。比在
训练日志里 grep `rollout/raw_reward` 可靠：日志会被 Ray 的多进程输出打断，JSONL 不会。

判据用窗口均值而不是单步——单步噪声很大，v1 的曲线单步能从 0.539 掉到 0.397。

用法：

    python3 omni_s2tt/curve.py /data/s2tt/ckpt/s2tt_run1/rollout_result/train
    python3 omni_s2tt/curve.py <上面那个目录> --csv curve.csv
"""

from __future__ import annotations

import argparse
import json
import pathlib


def load_steps(d: pathlib.Path) -> list[tuple[int, list[float]]]:
    steps = []
    for p in sorted(d.glob("*.jsonl"), key=lambda x: int(x.stem) if x.stem.isdigit() else -1):
        rewards = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line).get("reward")
                if isinstance(r, (int, float)):
                    rewards.append(float(r))
        if rewards:
            steps.append((int(p.stem) if p.stem.isdigit() else len(steps), rewards))
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="{--save}/rollout_result/train")
    ap.add_argument("--csv", default="", help="另存一份 step,mean,min,max,n")
    ap.add_argument("--window", type=int, default=10, help="首尾窗口大小")
    args = ap.parse_args()

    d = pathlib.Path(args.result_dir)
    if not d.is_dir():
        raise SystemExit(f"没有这个目录：{d}")

    steps = load_steps(d)
    if not steps:
        raise SystemExit(f"{d} 里没有可解析的 JSONL —— 训练是不是没设 --save / --rollout-result-dir？")

    rows = []
    for step, rewards in steps:
        mean = sum(rewards) / len(rewards)
        rows.append((step, mean, min(rewards), max(rewards), len(rewards)))
        print(f"  step {step:3d}: BLEU={mean:.3f} |{'#' * int(round(mean * 40))}", flush=True)

    means = [m for _, m, _, _, _ in rows]
    k = min(args.window, max(1, len(means) // 3))
    first, last = sum(means[:k]) / k, sum(means[-k:]) / k
    print(f"\n  前 {k} 步均值 = {first:.3f}")
    print(f"  后 {k} 步均值 = {last:.3f}")
    print(f"  区间 [{min(means):.3f}, {max(means):.3f}]，共 {len(means)} 步")
    print("  v1 参照：前 10 步约 0.29，31-40 步约 0.41")
    print(f"  判定：{'上升' if last > first else '没升'}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            f.write("step,mean,min,max,n\n")
            for step, mean, lo, hi, n in rows:
                f.write(f"{step},{mean:.6f},{lo:.6f},{hi:.6f},{n}\n")
        print(f"\n  已写 {args.csv}")


if __name__ == "__main__":
    main()
