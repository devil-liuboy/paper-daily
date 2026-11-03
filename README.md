# boy的论文日报

基于 [Futuresxy/paper-daily](https://github.com/Futuresxy/paper-daily) 定制：自动检索 arXiv，用 DeepSeek 生成中文技术分析，再由 GitHub Pages 展示。

本说明对应个人定制版。上游原始说明保留在 [UPSTREAM_README.md](UPSTREAM_README.md)，其中的默认模型、运行时间和会议来源不适用于本版。

部署后的预期网址为 <https://devil-liuboy.github.io/paper-daily/>。**实际是否上线，以仓库 Settings → Pages 给出的地址及成功部署记录为准；本说明不代表云端验收已完成。**

## 日常怎么看

打开网站后，可按日期、方向、匹配度和关键词筛选，查看问题、方法、创新、证据、局限以及原文/PDF 链接。

中文分析基于标题和摘要，没有阅读全文。实验数字属于作者报告，未独立复现；high / medium / low 表示与研究方向的匹配程度，不是论文质量评级。每篇会标明模型摘要、缓存复用或基础信息/失败状态；没有 PDF 的记录不会把其他链接冒充 PDF。

## 已配置的研究方向

| 方向 | 优先级 | 单次新增上限 |
| --- | ---: | ---: |
| GUI Agent 与计算机操作 | 2.0，第一优先 | 受总上限约束 |
| 多模态推理效率与视觉 Token 压缩 | 1.8 | 受总上限约束 |
| VLM 与多模态理解 | 1.6 | 受总上限约束 |
| 上下文与 KV Cache 压缩 | 1.6 | 受总上限约束 |
| 长任务压缩与 Agent 记忆 | 1.6 | 受总上限约束 |
| 跨领域 AI 探索 | 0.65 | 2 篇 |
| 地下退化环境导航 | 0.25 | 1 篇 |

优先级用于选稿排序，不会直接提高论文的相关性分数。只启用 arXiv；上游的体系结构会议池已关闭。每方向最多抓取 50 条候选，并非完整覆盖所有相关论文。没有合适新稿时不强行补旧论文凑数。

## 首次部署

1. 把本定制版代码上传到自己公开仓库的 `main` 分支。仅 Fork 上游不会包含这些定制。
2. 在 **Settings → Secrets and variables → Actions → Secrets → New repository secret** 添加 `DEEPSEEK_API_KEY`，值填写自己的 DeepSeek Key。不要放到代码、Issue 或普通 Variables 中。
3. 在 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
4. 若 Fork 后 Actions 尚未启用，在仓库 **Actions** 页面启用工作流。
5. 打开 **Actions → Paper Daily → Run workflow**，分支选择 `main`。首次推荐 `lookback_days=7`、`max_summaries=15`、`initialize_budget=true`。初始化只创建缺失账本，不覆盖已有账本。
6. 检查采集、摘要及 Deploy 步骤，随后用 Settings → Pages 的实际网址打开网站。不能只看进程退出码或网页能否打开，还应核对摘要成功/失败数和论文链接。

`max_summaries=1` 仅用于连通测试：它仍可能收录 15 篇论文，但只为其中一篇生成模型摘要。普通首次运行建议直接选 `15`，避免留下待补全摘要。多次手动运行共享当天 15 次付费尝试额度，不会因重跑而重置；新稿优先使用额度，历史补全取决于剩余额度。

## 自动运行与手动更新

- 正常每天北京时间 **09:00** 调度一次。GitHub Actions 可能延迟，不能保证九点整更新完成。
- 每次最多新增 **15 篇**；最多生成 **15 篇**模型摘要。付费尝试另有北京时间每日总上限 15 次，包括失败尝试和手动重跑。
- 定时运行使用上次成功记录，并保留 24 小时重叠窗口；手动运行按所选回看天数查询。重叠论文不占新稿名额，摘要按内容指纹复用。
- 存储上限为 900 篇，保留策略使用最近 60 天窗口，旧的高/中匹配论文可能继续保留；这不是完整论文数据库。
- 之后手动运行通常选择 `lookback_days=2`、`max_summaries=15`、`initialize_budget=false`。
- 暂停自动更新：打开 **Actions → Paper Daily → 右侧菜单 → Disable workflow**。恢复时选择 Enable workflow。禁用后原网页仍可访问，只是不再更新。

## 修改研究方向

可直接编辑 `config/interests.json`。也可以由仓库所有者创建一个标题为 **Research Interests** 的公开 Issue，使用仓库提供的模板，只编辑 JSON 中的 `topics`。

Issue 配置优先于文件配置；需要恢复文件配置时，关闭对应 Issue。建议只保留一个打开的配置 Issue。只有仓库所有者的配置生效，其他账号不能通过 Issue 修改研究方向或触发这个工作流。Issue 不能修改来源、API 地址或密钥。

每个主题使用 `id`、`name`、`description`、`keywords`、`arxiv_categories`、`priority` 和 `max_daily`。`max_daily=0` 表示没有该主题的独立上限，仍受总上限约束；当前检索使用每主题前 8 个关键词。研究兴趣和日报网页均公开，不要在配置中填写敏感内容。

## DeepSeek 与费用

仅调用 DeepSeek 官方接口，固定模型 `deepseek-flash`，关闭思考模式，每次输出上限 1800 tokens，不回退到 OpenAI。

程序月预算为 **20 元**，按北京时间自然月统计。当前配置按输入未命中缓存 **2 元/百万 tokens**、输出 **8 元/百万 tokens** 的高峰价格保守估算，不扣除缓存或低峰折扣。价格依据：[DeepSeek 官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。价格变化时应先核对并更新工作流中的费率。

每次调用前，先把最大预计费用预留到 `paper-state` 分支的 `usage.json`；成功后按 API 返回用量估算结算。失败或计费不确定时保留全额预留，不自动重试。预算不足、账本缺失/损坏或远程持久化失败时，拒绝新的付费调用。

**20 元是本程序按配置价格执行的限制，不是 DeepSeek 账户侧硬消费封顶。** 同一 Key 在其他程序里的消费不在此账本内，实际账单以 DeepSeek 平台为准。

## 数据保存与排错

`main` 保存程序与配置；`paper-state` 保存论文元数据、中文摘要和费用账本，不保存密钥、完整提示词或原始 HTTP 响应。**不要删除、回滚或手动清空 `paper-state` 分支中的账本来“重置预算”。**

| 现象 | 检查位置或处理方式 |
| --- | --- |
| 网站不更新 | 查看 Actions 最近一次 Paper Daily 运行及失败步骤 |
| 全部来源失败 | 保留旧数据且不推进采集水位；查看 arXiv 请求错误，稍后再试 |
| 摘要失败，但仍能看到论文 | 查看网页状态和工作流的 Report summary failures；检查 Secret、余额、预算或 API 响应 |
| 账本缺失或冲突 | 恢复原账本/状态分支后重试，不重新初始化来绕过历史消费 |
| 预算或每日调用次数用完 | 当次不再付费生成摘要；等待下个预算周期或核查后调整程序配置 |
| 首次看到空页面 | 主分支数据是“尚未初始化”的空数据；需完成一次成功采集与部署 |

本地预览无需调用模型，在仓库目录运行 `python -m http.server 8000 --directory web`，再访问 <http://localhost:8000>。主分支的初始空数据不是云端运行结果；最新论文在发布的网站和 `paper-state` 分支中。
