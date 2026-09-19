# 服务器源码同步说明（2026-09-19）

这是实际服务器部署后修改过的完整源码，不是线上数据备份，也不是 Git 仓库镜像。服务器目录由原始部署 ZIP 解压而来，没有 `.git`；不要把原包记录的提交号误认为这些新增修改已经提交到了 GitHub。

## 包含什么

- GPT-5.5 API 图片理解、DOCX 图片与正文生成知识语义卡；客户截图用于问答，不自动入库。API-only 模式关闭所有本地 OCR 回退。
- 低内存 Compose 覆盖、分阶段模型准备、AVX2 量化与历史 CPU OCR 排障代码。
- DOCX 依赖补齐、内置文档管理重索修复、OpenSearch 启动重试、管理页面说明修正及回归测试。
- 管理登录防爆破、Cloudflare 真实来源 IP、安全巡检、1Panel 证书同步及服务器验收脚本。
- 服务器部署记录、原项目自带的两份 `resources/knowledge/` 知识资料和前端资源。
- 本次交付补充了安全打包、忽略规则和同步说明；没有改动或重建正在运行的应用。

## 明确不包含

生产 `.env.production`、本地 `.env`、管理员密码、模型 API 密钥、私钥/证书、验收状态与截图、服务器回滚目录、会话或知识库数据库、线上上传资料、Docker 镜像/卷、模型权重、虚拟环境、node_modules、构建缓存和 Git 历史。

原项目自带知识资料仍属于你的业务内容；仓库若是公开的，请先审核这些资料。部署文档和 `infra/deploy/` 包含本服务器域名、路径及既有服务名称，虽然不是密钥，也不要未经调整就用于其他服务器。特别不要直接把本机 `server-security-check` 覆盖到其他服务器。

## 同步本地与 GitHub

1. 把完整 ZIP 解压到一个新目录，先对比，不要直接覆盖有未提交修改的本地仓库。
2. 在本地原仓库先运行 `git status`，备份或提交自己的改动，再创建同步分支，例如 `git switch -c sync/server-20260919`。
3. 对照同批交付的 `CHANGES.md` 合并新增/修改文件。不要删除本地 `.git`、私密配置、模型和 `data/`；这些内容未包含在包里不代表应从本地删除。
4. `server-changes.patch` 的基线是你最初上传的 `ai-customer-service-server.zip` 的文件内容，而不是仅凭提交号生成的补丁。若本地文件与该包一致，可先执行 `git apply --check /path/to/server-changes.patch`，检查成功后再 `git apply /path/to/server-changes.patch`。如有冲突、新增文件已存在或换行差异，先停下，按变更清单合并，不要强制覆盖。不要把完整包覆盖和补丁应用重复执行。
5. 合并后检查 `git diff`、`git diff --check` 与待提交文件，运行相应测试，确认没有密钥、数据库或本地回滚目录，再自行提交和推送 GitHub。本次服务器操作没有替你创建提交或推送任何远端。

完整 ZIP 内的 `RELEASE.json` 和 `SHA256SUMS` 是本次导出生成的元数据；原始上传 ZIP 没有被修改。交付目录的 `.zip.sha256` 用于校验整个完整包。

## 本地运行与后续更新的注意事项

- 当前线上方案请看 `API_ONLY_DEPLOYMENT.md`。服务器运行时必须同时使用 `compose.deploy.yaml` 与 `compose.low-memory.yaml`；单独使用原始 Compose 或根目录 `.env.example` 不等同于已部署配置。
- 本地想复现 API-only 图片模式，要在自己私密的配置中设置 `IMAGE_FEATURES_ENABLED=1`、`IMAGE_SEMANTIC_ENABLED=1`、`LOCAL_OCR_ENABLED=0`、`PRELOAD_OCR_MODEL=0`，并按自身 CPU 选择 AVX2/AVX512 文件；不要复制服务器密钥。
- 按用户要求，原始 `ai-customer-service-backend:packaged` 镜像已删除。历史 `backend-runtime-patch.Dockerfile` 仍引用它，不能直接用于下一次快捷更新；用户明确将更新脚本的处理留到以后，本次没有修复或重新构建它。普通 `backend.Dockerfile` 保留完整构建方式。
- 证书、代理和巡检配置是源码副本，不会因把文件同步到 GitHub 就自动安装到另一台机器。
- `SERVER_DEPLOYMENT.md` 与 `DEPLOYMENT_CHECKS.md` 保留历史阶段的记录；当前服务器结果看 `API_ONLY_DEPLOYMENT.md`。本次源码打包检查不等于重新执行全部付费模型、浏览器或新服务器部署验收。

## 重新打包

有 Git 工作区时使用 `python scripts/package_server.py`。仅有解压源码时可显式使用：

```bash
python scripts/package_server.py --allow-unversioned --output output/source-sync --archive-name ai-customer-service-sync.zip
```

打包脚本使用明确的源码允许清单，生成逐文件 SHA-256，并重新打开 ZIP 校验 CRC 与内容。仍应在对外共享前检查自己的源码中是否存在误写的凭据。
