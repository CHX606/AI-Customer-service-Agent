# OpenSearch 迁移

当前检索链路已经从“本地 Chroma + 内存 BM25”切换为 OpenSearch：

- `content`：原始分块正文。
- `content_terms`：jieba 搜索模式预分词，由 OpenSearch BM25 持久化索引。
- `embedding`：BGE 512 维向量，使用 Lucene HNSW 索引。
- `tenant_id`：所有查询、删除和重建操作的强制过滤字段。
- 搜索管道：OpenSearch 服务端融合 BM25 与向量排名，使用 RRF。

SQLite 与 OpenSearch 的职责不同：SQLite 保存租户、知识源、索引状态和缓存元数据；
OpenSearch 保存可检索的知识块及其关键词、向量和租户隔离字段。两者都需要保留。
Compose 文件固定使用 `aicustomerserviceagent` 项目名，确保从不同目录执行命令时
仍复用同一个容器和 `aicustomerserviceagent_opensearch-data` 数据卷。

## 本地启动

复制 `infra/opensearch/.env.example` 为仅本机使用的配置文件，替换其中密码，然后执行：

```powershell
docker compose --env-file infra/opensearch/.env.local `
  -f infra/opensearch/compose.yml up -d
```

把同一组 `OPENSEARCH_*` 应用连接变量加入项目 `.env`。本地自签名证书可设置
`OPENSEARCH_VERIFY_CERTS=0`；生产环境必须开启证书校验并配置 CA。

## 重建索引

OpenSearch 就绪后，从原始知识文件重新解析、切块和生成向量：

```powershell
.\.venv\Scripts\python.exe -m scripts.migrate_to_opensearch --tenant default
```

多个租户可以重复传入 `--tenant`。旧 `data/chroma_db` 不会自动删除，可在完成
检索验收和备份后再人工归档。

应用 `.env` 的 `OPENSEARCH_PASSWORD` 必须与 Compose 配置中的
`OPENSEARCH_INITIAL_ADMIN_PASSWORD` 一致。应用启动时会检查索引、别名和搜索管道；
文档上传或删除则通过统一索引服务增量同步到 OpenSearch。

## 生产要求

本地 Compose 只用于开发验证。生产环境应使用多节点 OpenSearch 或托管服务，
配置 TLS、密钥管理、专用主节点、数据节点、副本、快照仓库、监控告警和容量规划。
索引升级时增加 `OPENSEARCH_INDEX_VERSION`，完成全量重建和验收后再切换别名。
