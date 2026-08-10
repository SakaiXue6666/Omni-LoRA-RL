# v1 存档 patch

三个 fork 相对各自**上游基线**的完整改动。用途是**兜底**：即使 fork 仓库丢失/私有化/被误删，
只要有上游仓库和这三个文件，就能还原出 v1 的代码。日常仍以 submodule 为准，不要用 patch 覆盖。

导出于 2026-08-10，对应 `README_v1.md` 第 2 节的冻结点。

| patch | 上游仓库 | 上游基线（merge-base） | fork 快照 | 规模 |
|---|---|---|---|---|
| `relax.patch` | `redai-infra/Relax` | `01973f3a` | `6cde0798` | 33 文件 +4723/-61 |
| `sglang.patch` | `sgl-project/sglang` | `19b60a4f9` | `d13903a9` | 7 文件 +652/-16 |
| `sglang-omni.patch` | `sgl-project/sglang-omni` | `5cefa39e` | `185d6526` | 26 文件 +2187/-55 |

基线由 `git merge-base HEAD origin/main` 实测得到，与 `IMPORTANT/DIFF_SUMMARY.md` 第 1 节
记录的 Relax / sglang 基线一致；`sglang-omni` 的基线是本次补测的。

## 还原

```bash
git clone https://github.com/redai-infra/Relax.git
cd Relax && git checkout 01973f3a && git apply ../patches/relax.patch

git clone https://github.com/sgl-project/sglang.git
cd sglang && git checkout 19b60a4f9 && git apply ../patches/sglang.patch

git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni && git checkout 5cefa39e && git apply ../patches/sglang-omni.patch
```

## 重新导出 / 校验

```bash
# 重新导出（改了 fork 代码后）
git -C Relax       diff --binary --output=../patches/relax.patch       01973f3a..HEAD
git -C sglang      diff --binary --output=../patches/sglang.patch      19b60a4f9..HEAD
git -C sglang-omni diff --binary --output=../patches/sglang-omni.patch 5cefa39e..HEAD

# 校验：能逆向套用到当前工作树，说明 patch 内容与 base..HEAD 完全一致
git -C Relax       apply --check --reverse ../patches/relax.patch
git -C sglang      apply --check --reverse ../patches/sglang.patch
git -C sglang-omni apply --check --reverse ../patches/sglang-omni.patch
```

> 注意 `git apply` 不带提交历史；要保留逐次提交请用 fork 仓库本身或 `git format-patch`。
