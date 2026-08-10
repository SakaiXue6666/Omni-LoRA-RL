"""确认候选 slime tag 与 v1 缓存镜像版本一致（终局判定）。

链条回顾：
    modal_v1_probe.py  -> Modal 缓存的 slimerl/slime:latest 报出 v1 真实版本：
        torch 2.9.1+cu129 / transformers 4.57.1 / flashinfer{,-cubin} 0.6.3
        megatron-core 0.16.0rc0 / megatron-bridge 0.3.0rc0 / TE 2.10.0 / sglang 0.5.9
        dist-packages mtime = 2026-02-23T12:03:37Z（基础层 lmsysorg/sglang 的时间戳）
    slime_tag_lookup.py -> 二月窗口只有两个 tag：
        nightly-dev-20260210a  created 2026-02-10T15:16+08
        nightly-dev-20260225a  created 2026-02-25T19:03+08   <- 基础层 2/23 与之相容
    本脚本 -> 拉候选镜像，逐包比对，命中即可用 @sha256 冻结 v1。

用法：
    modal run modal_v1_confirm.py                       # 默认查 20260225a
    $env:V1_CANDIDATE="<digest 或 tag>"; modal run modal_v1_confirm.py
"""

from __future__ import annotations

import os

import modal

CANDIDATES = {
    "20260225a": "slimerl/slime@sha256:3fc40f5f1ea9805c7ef558dbc7f9ff093c6fa39796d1355d7e9a9b346b267ccf",
    "20260210a": "slimerl/slime@sha256:e4968bf2cfdeccbaecd0b0672e29d3fb741db17d3acb1b1e38841ffe85aaf509",
}
CANDIDATE = os.environ.get("V1_CANDIDATE", CANDIDATES["20260225a"])

# v1 缓存镜像的实测值，作为比对基准。
V1_EXPECTED = {
    "torch": "2.9.1+cu129",
    "transformers": "4.57.1",
    "megatron-core": "0.16.0rc0",
    "megatron-bridge": "0.3.0rc0",
    "transformer-engine": "2.10.0",
    "flashinfer-python": "0.6.3",
    "flashinfer-cubin": "0.6.3",
    "flashinfer-jit-cache": "0.6.3+cu129",
    "sglang": "0.5.9",
    "ray": "2.55.1",
    "numpy": "1.26.4",
}

_PROBE = r"""
import os, sys, time, glob
from importlib.metadata import version, PackageNotFoundError

EXPECTED = {expected!r}

print("PYTHON:", sys.version.replace("\n", " "), flush=True)
print("========== 与 v1 缓存镜像逐包比对 ==========", flush=True)
mismatch = []
for name, want in EXPECTED.items():
    try:
        got = version(name)
    except PackageNotFoundError:
        got = "<未安装>"
    mark = "OK " if got == want else "DIFF"
    if got != want:
        mismatch.append((name, want, got))
    print(f"  [{{mark}}] {{name}}: v1={{want}}  候选={{got}}", flush=True)

print("========== 构建时间线索（mtime）==========", flush=True)
for pat in ("/usr/local/lib/python3*/dist-packages/torch/version.py",
            "/usr/local/lib/python3*/dist-packages/transformers/__init__.py"):
    for p in glob.glob(pat):
        print(f"  {{p}}: {{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(os.path.getmtime(p)))}} UTC", flush=True)

print("\n========== 结论 ==========", flush=True)
if mismatch:
    print(f"  [MISMATCH] {{len(mismatch)}} 个包不一致: {{[m[0] for m in mismatch]}}", flush=True)
    sys.exit(1)
print("  [MATCH] 候选镜像与 v1 缓存镜像版本完全一致，可用其 digest 冻结 v1。", flush=True)
""".format(expected=V1_EXPECTED)

image = modal.Image.from_registry(CANDIDATE, add_python=None)

app = modal.App("v1-image-confirm")


@app.function(image=image, timeout=20 * 60)
def confirm() -> int:
    import subprocess
    import sys

    print(f"########## 候选: {CANDIDATE} ##########", flush=True)
    proc = subprocess.run([sys.executable, "-c", _PROBE], check=False)
    return proc.returncode


@app.local_entrypoint()
def main() -> None:
    rc = confirm.remote()
    if rc != 0:
        raise SystemExit("候选与 v1 不一致，换另一个候选再试")
