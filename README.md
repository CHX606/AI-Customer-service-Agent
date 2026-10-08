# AI Customer Service Agent

基于 FastAPI 与 LangGraph 的多租户 RAG 智能客服。结合 OpenSearch 混合检索、CPU 精排与证据约束回答，支持文档和图片知识、多轮咨询，以及用户提交人工处理申请后的邮件通知和后台跟进。

## 核心能力

| 能力 | 实现 |
| --- | --- |
| 混合检索 | OpenSearch 持久化 BM25 与 HNSW 向量索引，通过 RRF 融合召回；多个改写查询再次融合，检索按租户字段过滤 |
| CPU 精排 | CrossEncoder 对候选资料精排，提供 ONNX INT8 推理配置；有界信号量限制同时推理数，超额请求排队并设置等待超时 |
| 分级路由 | 明确的单轮问题优先走确定性规则，其余单轮问题使用轻量结构化路由；多轮保留问题状态和上下文分析，调用失败时按路径回退 |
| 证据约束回答 | 在一次结构化模型调用中判断证据状态并生成答复；仅覆盖部分问题时回答有依据的部分，信息不足时追问，未命中或冲突时提示人工处理 |
| 图文知识 | 支持 Word 图文知识入库和用户截图查询，可选择本地 OCR 或视觉模型 API；为文本命中块补齐同源、同版本、同章节的相邻内容 |
| 人工处理申请 | 用户主动提交后保存事项与会话快照，通过 SMTP 通知；后台记录处理状态和备注，同一提交编号保持幂等 |
| 多租户与一致性 | 会话按租户与会话编号隔离，以版本检查避免覆盖新回复；知识更新使缓存失效，写缓存时在事务内再次校验版本 |
| 前端与部署 | React / TypeScript 前端通过 NDJSON 接收状态和回答；提供 Docker Compose 标准与低内存部署配置 |

## 架构

```text
React 前端
    │ NDJSON 状态与回答
    ▼
FastAPI 接口层：认证、请求校验、协议转换
    │
    ▼
LangGraph 工作流
请求分析 → 路由 → 混合检索 → 精排与上下文扩展 → 回答 / 追问 / 人工处理提示
    │
    ├─ OpenSearch：知识正文、关键词和向量索引
    ├─ SQLite：租户、知识源、缓存元数据
    └─ SQLite 会话库：最近消息、活动问题、写入版本

用户提交人工处理申请 → SQLite 申请记录 → SMTP 通知 → 后台跟进
```

代码按 `domain / application / infrastructure / interfaces` 分层，依赖在 `bootstrap.py` 统一组装。详见 [工程分层说明](docs/architecture/engineering.md)。

## 关键设计

- **先规则、后模型**：明确请求由规则直接路由，复杂多轮保留上下文分析，减少简单请求的额外模型调用。
- **关键词与语义互补**：BM25 处理专有词与精确表达，向量召回覆盖口语化改写；RRF 按排名融合，之后用 CrossEncoder 精排。
- **部分回答与追问**：资料只覆盖部分问题时先回答有依据的内容，缺少关键情况时追问，并保留当前问题状态。
- **人工建议与申请分开**：聊天提示人工处理不会自动创建申请或发送邮件；用户明确提交后才登记，邮件失败不影响申请保存，通知状态可在后台查看。
- **缓存保持可选**：语义答案缓存默认关闭；启用后仅缓存满足条件的首轮问题，知识更新和事务内版本校验用于避免旧答案再次写回。

知识库正文流式输出在存在候选资料且模型声明可完整或部分回答时开放，最终结果校验结构字段。这些检查不等于自动验证答案事实正确，效果仍需固定测试集和人工复核。

## 评测记录

以下为已有本机实验记录，本次文档修改未重新运行；不代表当前服务器或线上流量指标。

| 测试 | 条件与记录 |
| --- | --- |
| 检索与精排 | Ryzen 7 7840H、ONNX AVX512 INT8、8 个候选；45 条可评分问题重复 3 轮，共 135 个样本，Hit@5 91.11%、MRR 0.8063，精排 P95 约 1.09 秒 |
| 端到端回答 | 完整测试集 63 条；并发 1 / 5 / 10 的完整回答 P95 为 7.12 / 9.37 / 11.61 秒，规则评分通过率均为 80.95% |
| 有依据回答对照 | 2026-09-28，29 条用例，优化前后各运行 3 遍；同一知识库快照、模型配置和单并发下，部分回答用例由 1/6 提升至 6/6，先追问再提示人工处理的流程由 0/9 提升至 7/9 |

规则评分依据关键词、路由与回答行为，不等于事实正确率。历史对照仍发现内部操作说明等回答问题，有限样本的改善不能推导为线上整体质量结论。重排并发上限按部署配置调整，当前配置样例为 2；历史本机 AVX512 数据不能直接代替服务器 AVX2 配置的性能。

测试命令、口径和已知限制见 [性能测试](docs/testing/performance.md) 与 [有依据回答评测](docs/testing/grounded-optimization.md)。

## 技术栈

Python 3.12 · FastAPI · LangGraph / LangChain · OpenSearch · sentence-transformers · ONNX Runtime · SQLite · React · TypeScript · Vite / Vitest · Docker Compose

## 文档

| 主题 | 文档 |
| --- | --- |
| 工程分层 | [工程分层说明](docs/architecture/engineering.md) |
| 性能测试 | [性能测试说明](docs/testing/performance.md) |
| 有依据回答 | [评测与限制](docs/testing/grounded-optimization.md) |
| 人工处理 | [邮件转人工流程](docs/setup/email-handoff.md) |
| OpenSearch | [迁移说明](docs/migrations/opensearch.md) |
| Windows 环境 | [CPU 环境说明](docs/setup/cpu-windows.md) |
| Linux 部署 | [部署说明](DEPLOY.md) |

## 目录

```text
back/
  domain/                # 业务模型、会话契约与错误
  application/           # 聊天、图片、企业资料、知识源用例和抽象接口
  infrastructure/        # SQLite、OpenSearch、模型与文件适配器
  bootstrap.py           # 依赖组装
  agent/                 # 请求分析、工作流、证据约束回答
  core/                  # LLM 与项目级路径配置
  interfaces/http/       # 接口网关、认证、协议转换
  knowledge/
    ingestion/           # 文档加载、DOCX 解析与切分
    images/              # 图片语义、OCR 与图片查询
    indexing/            # 索引编排
    retrieval/           # 混合召回、精排与缓存策略
  tenant/                # 租户初始化及兼容入口
evaluation/              # 基准程序、测试集和结果
tests/                   # 离线自动化测试
scripts/                 # 迁移与运维脚本
resources/knowledge/     # 内置知识源
infra/opensearch/        # 本地 OpenSearch 基础设施
docs/                    # 架构、部署、迁移与测试文档
front/                   # React 前端
```

分层职责、抽象接口、故障处理和会话持久化约定见 [工程分层说明](docs/architecture/engineering.md)。

## 本地启动

环境要求：Python 3.12、Node.js 24 和 Docker Compose。在项目根目录创建虚拟环境并复制配置：

```powershell
py -3.12 -m venv .venv
Copy-Item .env.example .env
Copy-Item infra\opensearch\.env.example infra\opensearch\.env.local
```

填写两个环境文件中的模型配置和 OpenSearch 密码后，安装依赖并启动后端：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m scripts.export_reranker_onnx --variant all --quantization avx512
docker compose --env-file infra\opensearch\.env.local -f infra\opensearch\compose.yml up -d
.\.venv\Scripts\python.exe -m scripts.migrate_to_opensearch --tenant default
.\.venv\Scripts\python.exe -m uvicorn back.api:app --host 127.0.0.1 --port 8000
```

ONNX 模型保存在被 Git 忽略的 `data/models/`；不支持 AVX512 的 CPU 将导出参数改为 `avx2`，并同步修改 `.env` 中的文件名。`back.api:app` 是稳定兼容入口，HTTP 实现在 `back/interfaces/http/`。

`.env` 中的 `OPENSEARCH_PASSWORD` 必须与
`infra/opensearch/.env.local` 中的 `OPENSEARCH_INITIAL_ADMIN_PASSWORD` 一致。
复杂多轮分析默认允许两次连接重试；重试仍失败时会使用已保存的会话上下文降级，避免返回空白答案。

在另一个终端启动前端：

```powershell
Set-Location front
npm ci
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。前端配置和测试说明见 [前端文档](front/README.md)。

## 人工处理申请

用户点击“提交人工处理”填写事情后，申请和最近会话保存在数据库，服务发送邮件通知站长。管理后台的“人工待办”可查看问题、会话和通知状态，并记录处理状态及备注。仅提交申请才会发信；普通聊天不会自动创建申请。

本地 `.env` 或服务器 `.env.production` 填写 SMTP 配置。Gmail 使用 `smtp.gmail.com`、587、`starttls`，`SMTP_PASSWORD` 填写应用专用密码；收件人填入 `HANDOFF_NOTIFICATION_TO`。其他租户需要独立配置 `HANDOFF_NOTIFICATION_RECIPIENTS`。`HANDOFF_ADMIN_URL` 可填写实际后台地址，配置仅存在服务端。重启后生效。完整说明见 [邮件转人工流程](docs/setup/email-handoff.md)。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest --run-live-model -m live_model -q
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --suite full --dry-run
```

性能测试和生产迁移细节见 `docs/testing/performance.md` 与 `docs/migrations/opensearch.md`。

默认测试为离线回归；带 `--run-live-model` 的命令显式调用真实模型。会话库默认保存在 `data/sessions.db`，可通过 `SESSION_DB_PATH` 调整。

仓库仅保留源码、测试、文档及内置知识源；本地密钥、数据库、模型、上传文件、依赖目录和构建产物由忽略规则排除。新克隆的仓库无需已有 `data/app.db` 即可运行离线测试。
