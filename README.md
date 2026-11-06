# cann-ops

[![sim](https://github.com/BTXeplore/cann-ops/actions/workflows/sim.yml/badge.svg)](https://github.com/BTXeplore/cann-ops/actions/workflows/sim.yml)

昇腾 CANN Judge 算子题解的本地工作区 + 命令行工具。

- **cannjudge CLI**：一条命令拉题、提交、等评测，复用浏览器已登录的会话（不存 token/密码）。
- **题解目录**：每道题一个目录，`kernel.asc` 就是要提交的文件，其余是模板参考与自测脚本。
- **可验证**：`sim/` 用 Python 1:1 复刻 kernel 的堆语义与 golden 对拍，GitHub Actions 每次 push 自动跑（不需要 NPU）。

## 成绩

`vector/topkpqdistance`：**15/15 用例 Pass，排行榜 #1，score 98.88，15 个用例合计 168 ms**
（同题此前最好成绩 3.42 s，慢 20 倍）。

## 快速开始

前置：
- Node.js ≥ 20
- Edge 里打开过 <https://cannjudge.cn> 并已登录（CLI 走浏览器会话，首次会自己拉起 CDP daemon）

```bash
cd cann-ops

node tools/cannjudge.mjs whoami                 # 确认登录态
node tools/cannjudge.mjs problems vector        # 看某题库下的题目
node tools/cannjudge.mjs fetch vector/acoshgrad # 新题：拉模板 + 生成 meta.json
node tools/cannjudge.mjs submit problems/vector/acoshgrad --watch
```

`whoami` 第一次运行会自动 `spawn` 一个 CDP daemon（`tools/cdp-daemon.mjs`，监听 127.0.0.1:9400），
之后所有命令复用它。

## 命令

| 命令 | 作用 |
|---|---|
| `whoami` | 当前登录用户 |
| `contests` | 列出公共题库（token） |
| `problems <contest>` | 列题目（token + problemId） |
| `fetch <contest>/<problem> [dir]` | 拉模板：`ref/` 存原始模板，可编辑文件放目录根；生成 `meta.json` |
| `submit <dir> [--watch]` | 提交目录下所有 `.asc` / `.h`，`--watch` 等评测结束并打印每个用例耗时 |
| `status <submissionId>` | 查一次结果 |
| `open` | 在 Edge 里打开 cannjudge.cn（会切到前台） |

## 目录约定

```
problems/<contest>/<problem>/
├─ kernel.asc       ← 提交的文件（题目模板里 editable 的那几个）
├─ meta.json        ← problemId / contest / problem，submit 依赖它
├─ notes.md         ← 思路、踩坑、性能数据
├─ ref/             ← 原始模板（main.asc / run.sh / sample.py / reference.py …），只读参考
└─ sim/             ← 本地对拍脚本（Python 复刻 kernel 语义 + 与 golden 对比）
```

`submit` 只收集题目根目录下的 `.asc`/`.h`（自动跳过 `main.asc`、`judge.asc`、`data_utils.h`），
所以想拆多个头文件或 `.asc` 直接放根目录即可，平台侧支持最多 20 个用户文件。

## 为什么走浏览器 / 有没有坑

- 平台用 HttpOnly cookie 鉴权，CLI 拿不到 token，所以只能在页面上下文里 `fetch`（`tools/cdp.mjs`）。
- **后台标签页会被 Chrome 冻结**，此时 `Runtime.evaluate` 会挂住。CLI 的做法是先
  `Page.captureScreenshot` 截一帧把渲染进程唤醒，再重试（见 `cdp.mjs` 的 `evalPage`）。
- 判题机有提交频率限制（约 50 次/小时 + 每次间隔 ~3 分钟），`submit` 收到 429 会把
  `retry_after_seconds` 打出来。

## 发布到 GitHub

```bash
git init && git add -A && git commit -m "init"
# 在 GitHub 建空仓库后：
git remote add origin git@github.com:<user>/cann-ops.git   # 或 https://github.com/<user>/cann-ops.git
git push -u origin main
```

之后每次改完：

```bash
node tools/publish.mjs -m "perf(topkpqdistance): 单核 6 cycle/element"
```

## 题解

| 题目 | 状态 | 备注 |
|---|---|---|
| [`vector/topkpqdistance`](problems/vector/topkpqdistance/notes.md) | 15/15 Pass，score 98.88（#1） | 全局 top-k + 稳定次序，168 ms / 15 用例 |
