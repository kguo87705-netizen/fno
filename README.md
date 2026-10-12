# FNO / PINO 科研与 Obsidian 跨设备工作区

此私有仓库用于 Windows 与 Mac 之间同步文献阅读和科研工作。

## 同步范围

- `PINO超声成像知识库/`：知识库笔记、论文原文、附件与共享的 Obsidian 设置。
- `Research_Direction/`：研究路线、全部现有实验源代码、文献核查材料与 HDF5 数据。
- `.build/`：实验配置、协议、日志、指标、图表、审计和成本记录，以及辅助脚本。
- 根目录的共享 Obsidian 配置与项目迁移记录。

`.build/` 中包含科研证据，不应将整个目录忽略。

以下内容只保留在 Windows 本机，不随此仓库同步：所有 `.pt` 模型和张量文件、Windows Python 环境、Python/编译缓存、临时文件、重复的知识库 ZIP，以及每台设备自己的 Obsidian 窗口布局。完整原始文件的本地提交保存在 Windows 的 `codex/local-full-backup` 分支；不要将该分支推送到 GitHub。

## 在 Mac 上首次使用

安装 Git 与 Obsidian 后，在终端运行：

```sh
git clone https://github.com/kguo87705-netizen/fno.git
cd fno
```

私有仓库需要使用有访问权限的 GitHub 账号认证。不要将 GitHub 密码或访问令牌写入仓库。

在 Obsidian 中选择“打开文件夹作为仓库”，打开克隆得到的 `fno` 文件夹，即可同时看到知识库和研究目录。也可以单独打开 `PINO超声成像知识库` 子目录，只浏览该知识库。

Mac 上的项目根目录就是克隆得到的 `fno` 文件夹；无需创建 Windows 的 `C:` 或 `D:` 路径。

## 两台电脑之间的日常同步

每次开始工作前，在项目根目录运行：

```sh
git pull --ff-only
```

完成修改后，关闭正在编辑的文件，再运行：

```sh
git add --all
git commit -m "更新文献笔记与实验记录"
git push
```

没有修改时可以跳过提交。Git 不会自动同步；切换电脑前先推送，到另一台电脑后先拉取。尽量避免两台电脑同时修改同一篇笔记。若拉取提示分支分叉或文件冲突，先保留本地修改并解决冲突，不要用强制推送覆盖另一台电脑的工作。

Windows 上使用真实项目目录 `D:\下载\FNO_Research_Workspace` 执行这些命令。桌面旧入口的 Git 仓库是独立的空仓库，不用于本次同步。为使共享设置也保持一致，建议在 Windows Obsidian 中打开 D 盘真实项目目录。

## 在 Mac 上运行实验

同步文件并不意味着已完成实验环境迁移。请在 Mac 本机重新创建 Python 虚拟环境，并根据各实验目录的 README 和脚本安装依赖；Windows 的 `author-baseline-env` 不能直接复用。

现有日志、图表和指标可以直接查看。读取历史 `.pt`、恢复训练或某些复核脚本需要对应的本地大文件，应按需从 Windows 复制到同样的相对路径；这些文件仍由 `.gitignore` 排除。它们没有被删除，也不需要为了查看笔记重新运行实验。

部分辅助脚本仍含 Windows 绝对路径，运行前需改为 Mac 本机路径。实验结果中的历史路径保留原样。此次同步尚未在 Mac 上验证实验运行，也没有改动原有数值算法、停止规则或已有实验结论。
