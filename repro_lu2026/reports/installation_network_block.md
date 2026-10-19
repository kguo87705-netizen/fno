# 推送与 Deepwave 安装：网络阻塞记录

**历史记录：以下保存网络恢复前的失败状态。** 2026-10-19稍后用户确认网络恢复，原提交已推送，环境安装及GPU对比随后完成，最新状态见[Deepwave对比报告](deepwave_comparison.md)。保留本记录以追踪失败成本。

日期：2026-10-19（北京时间）。用户已批准原 26 个文件提交、推送及指定版本的 Deepwave 环境安装。

## 实际完成

原 26 个文件已提交到本地，提交号 `713fdc4d6d7ceec9f7c6ba4a0a98e440cda0d1c7`，提交身份为 `kguo87705-netizen <kguo87705@gmail.com>`，仅仓库级配置。提交未包含本报告和本轮 README 状态更新。

本地工作副本：`E:\FNO_research\repos\fno_light`。本轮新增目录仅在项目内，包括空的 `envs/cpython-3.12.15`；没有安装软件。下载/解压/安装成功不能从目录存在推断。

## 两次失败后停止

Git 推送第一次：`Recv failure: Connection was reset`。

Git 推送第二次：`Failed to connect to github.com port 443 after 21130 ms: Could not connect to server`。

便携 Python 3.12.15 下载第一次（PowerShell Invoke-WebRequest）：`Received an unexpected EOF or 0 bytes from the transport stream.`

下载第二次（curl HTTP/1.1）：`curl: (28) Failed to connect to github.com:443 after 21060 ms: Could not connect to server`。

第二次下载命令后续目录检查返回了成功退出码，但 curl 本身明确失败；没有把该命令组退出码误判为下载成功。未发现已完成的 Python 压缩包，目标解压目录为空。

判断：当前到 GitHub HTTPS 的连接失败，尚不能确定是网络、代理还是远端连接问题；没有证据表明是 Python 版本或 Deepwave GPU 兼容性问题。遵守同一问题两次失败即停，未继续切换下载源、安装或重试推送。

## 尚未完成

- 远程推送尚未成功，不能声称 GitHub README 页面已经更新。
- 未安装 Python、PyTorch、Deepwave 或其他依赖；未安装 CUDA Toolkit/编译器。
- 未执行 GPU 单炮验证、64 炮传播、原始或全局幅度/时移标定比较；不提供未经测量的 Deepwave 性能、显存或误差。
- 既有 k-Wave 测量和 300/500 样本估算保持有效，见[并行报告](concurrency.md)。Deepwave 尚无测量，不能据此推荐切换批量求解器。
- 未开始 M2。

本轮命令将 TEMP/TMP 指向项目 tmp 目录，文件操作均明确指向项目内；没有主动向 C 盘写入文件，也未开展操作系统级全盘写入审计。

网络恢复后应先完成已授权的原提交推送，再恢复 E 盘安装和比较。本报告与 README 的新增变更保留本地，等待下一轮清单确认，不随原提交推送。
