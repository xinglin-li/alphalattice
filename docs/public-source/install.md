# Install and start / 安装并启动

## Start AlphaLattice 0.2.10

After the [0.2.10 release](https://github.com/xinglin-li/alphalattice/releases/tag/v0.2.10)
is available, open a terminal in the folder where you want your workspace. Run the
command for your shell. It starts the local Host with the versioned release wheel
and its locked runtime packages. It uses the `alphalattice-workspace` and
`alphalattice-data` subfolders, so an older Host can keep serving on its own
machine data root. Open the Workbench
link printed by the Host. Keep the terminal running while you work. Use Ctrl+C to
stop the Host.

For follow-up commands keep the data-root setting and the launcher prefix through
`--workspace ./alphalattice-workspace` in this folder. See [Getting started](getting-started.md) for
system prerequisites and [Upgrading](upgrading.md) for existing workspaces.

## 启动 AlphaLattice 0.2.10

[0.2.10 发布](https://github.com/xinglin-li/alphalattice/releases/tag/v0.2.10)后，
在你选定的工作区目录中打开终端，执行适合当前终端的命令。它使用固定版本的发布 wheel
及其锁定的运行依赖来启动本地 Host，并使用 `alphalattice-workspace` 和
`alphalattice-data` 子目录。旧版 Host 可继续使用自己的机器数据根目录。打开 Host
打印的 Workbench 链接。使用期间保持
终端开启，按 Ctrl+C 停止 Host。

在同一目录中执行后续命令时，请沿用数据根设置及截至
`--workspace ./alphalattice-workspace` 的命令前缀。系统依赖见[入门](getting-started.md)，
已有工作区的处理方式见[升级](upgrading.md)。

### Windows PowerShell

```powershell
if (-not (Get-Command uvx -ErrorAction SilentlyContinue)) { throw 'Install uv / 安装 uv https://docs.astral.sh/uv/getting-started/installation/' } else { & { $priorRoot = $env:ALPHALATTICE_DATA_ROOT; try { $env:ALPHALATTICE_DATA_ROOT = Join-Path $PWD.Path 'alphalattice-data'; uvx --isolated --python 3.12 --from https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-0.2.10-py3-none-any.whl --with-requirements https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-runtime-requirements.txt --overrides https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-runtime-overrides.txt alphalattice --workspace ./alphalattice-workspace serve --no-browser } finally { $env:ALPHALATTICE_DATA_ROOT = $priorRoot } } }
```

### macOS and Linux / macOS 和 Linux

```bash
if command -v uvx >/dev/null 2>&1; then ALPHALATTICE_DATA_ROOT="$PWD/alphalattice-data" uvx --isolated --python 3.12 --from https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-0.2.10-py3-none-any.whl --with-requirements https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-runtime-requirements.txt --overrides https://github.com/xinglin-li/alphalattice/releases/download/v0.2.10/alphalattice-runtime-overrides.txt alphalattice --workspace ./alphalattice-workspace serve --no-browser; else printf '%s\n' 'Install uv / 安装 uv https://docs.astral.sh/uv/getting-started/installation/' >&2; false; fi
```

## First-use doctor

Run the [pinned wheel doctor](../../AGENTS.md#open-the-workspace) from the
launch folder. Use its `host_readiness` report to decide any repairs before
research. Run the same doctor again when a required dependency is missing.

In a fresh plugin folder the doctor reports `native_bridge.project_declaration_missing`
until you configure a project for Session binding. It still reports readiness.
For workspace commands, keep the explicit `--workspace` from the launch guide.

### 首次使用检查

请在启动目录运行[固定版本的 doctor 命令](../../AGENTS.md#open-the-workspace)。
研究前请根据 `host_readiness` 的报告处理缺失项，后续遇到缺少依赖时也可运行同一条检查命令。

首次使用插件时，在为会话绑定配置项目之前，doctor 会报告
`native_bridge.project_declaration_missing`，同时提供环境就绪报告。执行工作区命令时，
请沿用启动命令中的显式 `--workspace` 参数。

## Claude Code

Run this command in Claude Code 2.1.275 or later to add the AlphaLattice marketplace
and install the plugin. Then use `/alphalattice:start` in your chosen workspace. You can also give
your agent the [repository link and a research intent](getting-started.md).

在 Claude Code 2.1.275 或更新版本中执行以下命令来添加 AlphaLattice 市场并安装插件。然后在你选定的
工作区中运行 `/alphalattice:start`。也可以把[仓库链接和研究意图](getting-started.md)
直接交给智能体。

```text
/plugin install alphalattice --marketplace xinglin-li/alphalattice
```
