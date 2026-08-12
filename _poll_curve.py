"""每次轮询把新看到的步并进本地曲线文件。

`modal app logs` 只回吐尾部若干行，隔十分钟拉一次会漏掉中间的步。训练脚本结束时会
从容器内的完整日志打印整条曲线，那才是权威来源；这个文件只是防止中途崩了拿不到。
"""

import json
import os
import re
import subprocess
import sys

APP = "ap-t00uYovfitajPpXBRFaYbO"
MODAL = r"D:\Li_Lab\RL\.venv-modal\Scripts\modal.exe"
STORE = r"d:\Li_Lab\RL\omni-lora-rl-v2\_curve.json"

PAT = re.compile(
    r"rollout (\d+): \{'rollout/raw_reward': ([0-9.]+), 'rollout/response_lengths': ([0-9.]+)"
)

proc = subprocess.run([MODAL, "app", "logs", APP], capture_output=True, text=True, errors="replace")
text = proc.stdout + proc.stderr

curve = {}
if os.path.exists(STORE):
    with open(STORE, encoding="utf-8") as f:
        curve = json.load(f)

new = []
for m in PAT.finditer(text):
    step = m.group(1)
    if step not in curve:
        new.append(int(step))
    curve[step] = [round(float(m.group(2)), 4), round(float(m.group(3)), 1)]

with open(STORE, "w", encoding="utf-8") as f:
    json.dump(curve, f, indent=1, sort_keys=True)

steps = sorted(int(k) for k in curve)
print("已记录 " + str(len(steps)) + " 步，本次新增 " + str(sorted(new)))
print("  " + "  ".join(f"{s}:{curve[str(s)][0]}" for s in steps))
if steps:
    missing = [s for s in range(steps[0], steps[-1] + 1) if str(s) not in curve]
    if missing:
        print("  中间漏掉（滚出日志尾部）: " + str(missing))

bad = [l for l in text.splitlines() if re.search(r"RuntimeError:|CUDA out of memory|unable to open shared|preempt|Killed", l)]
print("异常: " + (bad[-1][:150] if bad else "无"))
sys.exit(0)
