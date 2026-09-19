# AI Customer Service Agent

面向生产演进的多租户智能客服：SQLite 保存租户、知识源和语义缓存元数据，OpenSearch 持久化 BM25 与向量索引，并通过 RRF 和 CPU Reranker 完成混合检索。

Linux 服务器全新部署见 [部署说明](DEPLOY.md)，交给服务器 Codex 的任务说明见 [CODEX_DEPLOY.md](CODEX_DEPLOY.md)。部署使用 `compose.deploy.yaml`，本地开发仍使用下方原有流程。

2026-09-19 服务器部署后的源码同步说明见 [SYNC_GUIDE.md](SYNC_GUIDE.md)。当前服务器使用 GPT-5.5 API 识图和 `compose.low-memory.yaml`，本地 OCR 关闭；当前验收与运维以 [API_ONLY_DEPLOYMENT.md](API_ONLY_DEPLOYMENT.md) 为准，原始部署文档中的全本地模型方案不代表当前服务器配置。

## 存储与检索架构

- SQLite：租户配置、知识源登记、索引状态和语义缓存元数据。
- 独立 SQLite 会话库：最近消息和当前问题状态持久化，按租户隔离并检查并发写入版本。
- OpenSearch：知识块正文、BM25 关键词索引、512 维向量索引和租户过滤字段。
- RRF：在 OpenSearch 搜索管道中融合 BM25 与向量召回结果。
- ONNX INT8 Reranker：对 8 个候选块精排，最终向回答模型提供 5 个证据块。

`data/chroma_db` 仅是迁移前遗留数据，当前运行链路不会读取它。

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
