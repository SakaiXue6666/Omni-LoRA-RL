"""把 v1 缓存镜像反查成一个具体的 slime tag + digest（本地跑，不动 Modal）。

背景：modal_v1_probe.py 已确认 v1 跑的是 Modal 于 2026-02-23 前后缓存的
`slimerl/slime:latest`（transformers 4.57.1 / flashinfer 0.6.3 / megatron-bridge 0.3.0rc0）。
但 `latest` 是可变 tag，要冻结 v1 必须拿到当时那份的 @sha256 digest。

做法：走 Docker Registry v2 只读元数据，不拉镜像（每个 tag 约 2 次 HTTP，几十 KB）。
    1. 匿名 token
    2. manifest（可能是 manifest list -> 取 amd64）
    3. config blob 里的 `created` = 镜像真实构建时间

判定：`created` 最接近 2026-02-23T12:03Z（v1 容器里 dist-packages 的 mtime）的那个 tag，
就是 v1 当时解析到的 latest。

用法：python slime_tag_lookup.py
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

REPO = "slimerl/slime"
TARGET = datetime(2026, 2, 23, 12, 3, 37, tzinfo=timezone.utc)
# 只查目标时间前后一段窗口内推送的 tag，避免 142 个全查。
WINDOW = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 4, 15, tzinfo=timezone.utc))

MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
    )
)


def _get_json(url: str, headers: dict[str, str] | None = None) -> tuple[dict, dict]:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode()), dict(resp.headers)


def list_tags() -> list[dict]:
    out: list[dict] = []
    url = f"https://hub.docker.com/v2/repositories/{REPO}/tags?page_size=100"
    while url:
        data, _ = _get_json(url)
        out.extend(data["results"])
        url = data.get("next")
    return out


def token() -> str:
    data, _ = _get_json(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    )
    return data["token"]


def created_of(tag: str, tok: str) -> tuple[str, str] | None:
    """返回 (镜像 digest, config.created)。"""
    hdr = {"Authorization": f"Bearer {tok}", "Accept": MANIFEST_ACCEPT}
    base = f"https://registry-1.docker.io/v2/{REPO}"
    try:
        manifest, headers = _get_json(f"{base}/manifests/{tag}", hdr)
    except Exception as e:  # noqa: BLE001
        print(f"  {tag}: manifest 获取失败 {e!r}", flush=True)
        return None

    digest = headers.get("Docker-Content-Digest", "")
    if "manifests" in manifest:  # manifest list / OCI index
        amd = [
            m
            for m in manifest["manifests"]
            if m.get("platform", {}).get("architecture") == "amd64"
            and m.get("platform", {}).get("os") == "linux"
        ]
        if not amd:
            return None
        digest = amd[0]["digest"]
        manifest, _ = _get_json(f"{base}/manifests/{digest}", hdr)

    cfg_digest = manifest.get("config", {}).get("digest")
    if not cfg_digest:
        return None
    cfg, _ = _get_json(f"{base}/blobs/{cfg_digest}", hdr)
    return digest, cfg.get("created", "?")


def main() -> None:
    tags = list_tags()
    print(f"仓库共 {len(tags)} 个 tag", flush=True)

    cands = []
    for t in tags:
        try:
            pushed = datetime.fromisoformat(t["last_updated"].replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            continue
        if WINDOW[0] <= pushed <= WINDOW[1]:
            cands.append((t["name"], pushed))
    cands.sort(key=lambda x: x[1])
    print(f"窗口 {WINDOW[0].date()}~{WINDOW[1].date()} 内 {len(cands)} 个候选\n", flush=True)

    tok = token()
    rows = []
    for name, pushed in cands:
        got = created_of(name, tok)
        if not got:
            continue
        digest, created = got
        try:
            cdt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            delta = abs((cdt - TARGET).total_seconds())
        except Exception:  # noqa: BLE001
            delta = float("inf")
        rows.append((delta, name, created, pushed.isoformat(), digest))
        print(f"  {name:<28} created={created:<32} pushed={pushed:%Y-%m-%d} {digest}", flush=True)

    rows.sort()
    print("\n========== 与 v1 mtime (2026-02-23T12:03:37Z) 最接近的 3 个 ==========", flush=True)
    for delta, name, created, pushed, digest in rows[:3]:
        print(f"  {name}  created={created}  相差 {delta/3600:.1f} 小时", flush=True)
        print(f"    {REPO}@{digest}", flush=True)


if __name__ == "__main__":
    main()
