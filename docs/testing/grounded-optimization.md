# 有依据回答优化的评测说明

## 已实施的行为

- 回答模型允许依据正文做一步推理，新增 `partial`，仅回答有依据的部分并说明需要人工确认的内容；仍禁止补充资料没有的金额、时限或处理方式。
- `sufficient` / `partial` 有回答且候选资料非空时，漏填或填错全部资料编号回填精排第 1 名。无候选资料或回答为空仍降级；冲突仍须两份有效资料。
- 两种可回答状态均支持流式输出，`partial` 不进入语义缓存。旧版独立证据判断保留原四状态。
- 同一问题首次 `not_found` 追问，第二次转人工；计数随会话持久化，继续与纠正保留计数。若上一条助手回复正是固定的未命中追问，即使重述被判为 `new_issue`，也按已追问一次处理；明确切换新问题时重新计数。更早的历史追问和用户自己粘贴的追问文案不会触发该兜底。
- 明确的人工客服请求使用现有企业资料规则直接路由。已有问题时，企业资料/人工客服插话保留原问题的摘要、意图、状态、追问及未命中计数，后续继续原问题；没有原问题的首次企业资料咨询保持原行为。
- 精排后补齐前后各一块，限制同租户、同来源、同内容版本、同章节；跳过图片语义块，合并正文最多 1400 字，扩展失败回退原结果。
- 新增判断状态、原因、精排分数与问题日志；应用默认启用自身 INFO 日志。

## 本次基线状态

基线提交为 `2cf5b27e019db817679041f0f4659b403fca7176`。初次离线检查时，本地服务不可用，因此当时没有真实模型对照结果。后续已恢复本地检索服务，并在 2026-09-28 完成下面的隔离在线评测。

`evaluation.benchmarks.multiturn --dry-run` 已通过，包含 8 条多轮用例。可用性记录保存在 `evaluation/results/grounded-optimization-before.json`；该文件的 `status=unavailable`，不应作为性能或质量基线。离线测试与真实模型评测分开验收。

## 2026-09-28 真实模型对照

使用用户提供的 `grounded_handoff.json` 和原评分脚本，29 条用例，改前、改后各运行 3 遍；每个版本 87 个会话、114 轮，合计 228 轮。两组使用同一份 59 块知识库快照、相同模型配置、单并发，关闭语义缓存。源码、应用数据库和会话数据库按版本隔离，未暂存或恢复工作区改动。

原始报告为 `evaluation/results/grounded-handoff-before.json`、`grounded-handoff-before.md`、`grounded-handoff-after.json` 和 `grounded-handoff-after.md`。配置及文件校验值记录在 `grounded-handoff-manifest.json`，对照解读见 `grounded-handoff-comparison.md`。

| 原评分脚本指标 | 改前 | 改后 |
|---|---:|---:|
| 应答题误转人工 | 3/75（4%） | 0/75（0%） |
| 应答题被归类为回答 | 66/75（88%） | 75/75（100%） |
| 部分回答用例通过 | 1/6 | 6/6 |
| 先追问再转人工流程符合 | 0/9 | 7/9 |
| 内部禁用词命中轮数 | 8 | 7 |
| 接口错误 / 语义缓存命中 | 0 / 0 | 0 / 0 |

这些指标不等于事实正确率。改后 iOS 软件推荐三次均未答出软件名；邮箱修改和返利提现仍把内部操作说明当成用户入口。机房信息的明确未知答复会被脚本归类为 `answer`，而“快捷入口”等变体又会漏过禁用词检查。当前结果能证明部分回答和追问流程改善，不能据此宣称内部信息控制已通过。原报告保留不改，另附逐轮 AI 复核记录供人工确认。

## 可复现的前后对照

离线回归使用以下命令。Windows 显式启用 UTF-8，避免已有图片缓存测试按系统 GBK 编码读取 UTF-8 文本而失败。

本次最终结果（包含重述追问兜底与原问题保留修正）：561 项通过、5 项真实模型测试跳过，另有 30 项子测试通过；2 项现有依赖弃用提醒。完整结果保存在 `evaluation/results/grounded-optimization-after-tests.xml`，汇总记录在 `evaluation/results/grounded-optimization-after.json`。该汇总不是在线效果评测结果。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest -q -p no:cacheprovider --junitxml=evaluation/results/grounded-optimization-after-tests.xml
```

多轮评测工具 `evaluation.benchmarks.multiturn` 会请求已经运行的 HTTP 服务 `/chat/stream`，并为每条用例创建新会话；它不会自动启动当前源码，也不保证正在运行的服务对应当前提交。

在专门的测试环境使用相同知识库快照、租户配置、模型版本、采样设置和并发度，分别运行优化前提交和优化后提交。启动时记录提交、配置及知识库版本，确认服务已加载对应源码，并将会话数据库与真实业务数据隔离。为了避免缓存影响，只比较相同缓存策略的结果。

```powershell
.\.venv\Scripts\python.exe -X utf8 -m evaluation.benchmarks.multiturn --base-url http://127.0.0.1:8000 --concurrency 1 --output evaluation/results/grounded-optimization-before-live.json
.\.venv\Scripts\python.exe -X utf8 -m evaluation.benchmarks.multiturn --base-url http://127.0.0.1:8000 --concurrency 1 --output evaluation/results/grounded-optimization-after-live.json
```

两条命令分别在对应版本的服务上运行。不要将端口连接失败产生的全失败报告当作优化前质量基线。需要租户令牌时使用 `BENCHMARK_TENANT_TOKEN` 环境变量，不写入报告。

## 指标边界与人工复核

现有工具记录首轮追问率、最终解决率、上下文保留率、完整闭环通过率和延迟。它使用关键词、路由和回答行为规则评分，不包含可靠的事实答错率或转人工率。

- `1 - success_rate` 是未拿到完整回答的会话占比，属于接口/运行错误指标，不是事实答错率。
- `1 - final_resolution_rate` 可能包含路由不符合预期、缺少关键词、接口失败等情况，不能换算成答错率。
- `context_retention_rate` 只检查测试集中列出的重复追问表达，不能覆盖所有上下文遗失。

人工复核应读取每条真实报告的 `samples[].turns[]`，逐轮对照完整会话、当时的知识库版本及支持资料，分别标注：

| 标注字段 | 取值 | 判定要求 |
|---|---|---|
| `handoff` | `yes` / `no` | 是否明确将当前问题交由人工处理；仅提及客服渠道不自动算转人工 |
| `factuality` | `correct` / `incorrect` / `not_applicable` / `unverifiable` | 回答是否包含与支持资料或已核实规则冲突的实质性事实；只有追问、转人工或接口失败时不作事实正确判断 |
| `review_reason` | 简短文字 | 记录转人工依据或具体错误、冲突资料编号 |
| `reviewer` | 复核人标识 | 使结论可追溯 |

按轮统计时，转人工率为 `handoff=yes` 的轮数除以成功返回最终回复的轮数；事实答错率为 `factuality=incorrect` 的轮数除以 `correct + incorrect` 的轮数。同步报告 `not_applicable`、`unverifiable` 和接口失败数量，避免通过少答或少评得到虚假的低错误率。若改用会话口径，必须单独注明，不能混用分母。

比较前后结果时，使用相同复核口径，并逐条查看新增错误、误转人工和成功补充信息的案例。有限测试集通过只能说明这些案例满足检查，不代表线上整体错误率已降低。
