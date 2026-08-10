"""v1 缓存镜像的完整清点：全量包清单 + 安装痕迹取证。

要解决的矛盾：
    modal_v1_probe.py 从 Modal 缓存的 slimerl/slime:latest 读到
        sglang 0.5.9   （PyPI 发布 2026-02-23，下一版 0.5.10 于 2026-04-05）
        ray    2.55.1  （PyPI 发布 2026-04-22）
    两者不可能自然共存于同一个 slime nightly：4 月下旬的 nightly 不会还带 2 月的 sglang。
    而实测 20260225a / 0226a / 0227a 的 ray 均为 2.54.0（2026-02-18 发布），其余 10 个包全一致。

    => 猜测 ray 是被某一层单独升级/替换的，而非镜像原生。本脚本取证：

  1. 全量 pip 清单（同时作为 README_v1 里 v1 环境的权威快照）
  2. 各 dist-info 的 mtime：若 ray 的时间戳明显晚于 torch/sglang，说明是后装的
  3. ray 的 INSTALLER / direct_url.json / WHEEL：看它从哪来（PyPI？wheel URL？）
  4. dist-packages 里 mtime 最新的若干条目：定位镜像最后一次改动发生在什么时候

用法：modal run modal_v1_deep.py
"""

from __future__ import annotations

import modal

# 必须与 v1 脚本里的镜像定义逐字一致，才能命中同一份 Modal 缓存。
V1_BASE = "slimerl/slime:latest"

_PROBE = r"""
import glob, os, sys, time
from importlib.metadata import distributions

SITE = "/usr/local/lib/python3.12/dist-packages"

def ts(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p)))

print("PYTHON:", sys.version.replace("\n", " "), flush=True)

print("\n========== 1. 全量包清单 ==========", flush=True)
rows = sorted(
    ((d.metadata["Name"] or "?", d.version) for d in distributions()),
    key=lambda x: x[0].lower(),
)
print(f"共 {len(rows)} 个发行包", flush=True)
for name, ver in rows:
    print(f"  {name}=={ver}", flush=True)

print("\n========== 2. 关键包 dist-info 的 mtime ==========", flush=True)
for key in ("ray-", "sglang", "torch-", "transformers-", "megatron", "flashinfer",
            "transformer_engine"):
    for p in sorted(glob.glob(f"{SITE}/{key}*.dist-info")) + sorted(glob.glob(f"{SITE}/{key}*.egg-link")):
        print(f"  {os.path.basename(p)}: {ts(p)} UTC", flush=True)

print("\n========== 3. ray 的安装来源 ==========", flush=True)
for p in glob.glob(f"{SITE}/ray-*.dist-info"):
    print(f"  {p}", flush=True)
    for fn in ("INSTALLER", "direct_url.json", "WHEEL", "REQUESTED"):
        fp = os.path.join(p, fn)
        if os.path.exists(fp):
            with open(fp) as f:
                print(f"    {fn}: {f.read().strip()[:400]}", flush=True)
        else:
            print(f"    {fn}: <无>", flush=True)

print("\n========== 4. dist-packages 中 mtime 最新的 25 项 ==========", flush=True)
entries = []
for name in os.listdir(SITE):
    p = os.path.join(SITE, name)
    try:
        entries.append((os.path.getmtime(p), name))
    except OSError:
        pass
entries.sort(reverse=True)
for m, name in entries[:25]:
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(m))} UTC  {name}", flush=True)
"""

image = modal.Image.from_registry(V1_BASE, add_python=None)

app = modal.App("v1-image-deep")


@app.function(image=image, timeout=20 * 60)
def deep() -> None:
    import subprocess
    import sys

    print(f"########## {V1_BASE}（Modal 缓存）##########", flush=True)
    subprocess.run([sys.executable, "-c", _PROBE], check=False)


@app.local_entrypoint()
def main() -> None:
    deep.remote()
