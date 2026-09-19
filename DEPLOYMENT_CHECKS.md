# 打包端验证记录

日期：2026-09-19。源码基线：`cdc78009a6dd6f6532dc32337f1fa86295928cd7`。ZIP 包含该基线当前工作区的部署新增文件，以包内 `SHA256SUMS` 为文件依据，不能只用 Git 提交号代替包的版本。

## 已通过

- 从部署 ZIP 解压出的源码运行后端离线测试，设置 `PYTHON_DOTENV_DISABLED=1`，禁止加载父目录的本机配置：**421 passed，5 skipped**。跳过的是显式要求真实模型的测试。
- 使用现有本地 Node 依赖，设置 `VITE_API_BASE_URL=/api`、`VITE_TENANT_ID=default` 后执行生产构建：通过；检查构建代码中的 API 配置为 `/api`。前端测试：**11 passed**。
- Python 运维脚本语法、Bash `preflight.sh` 语法和 `git diff --check`：通过。
- Compose CLI 解析完整部署配置：通过。验证模型/管理员必填配置，backend/OpenSearch 没有宿主机端口映射、管理员鉴权强制开启、两端 OpenSearch 密码一致、bcrypt 哈希经环境插值后保持一致。
- 官方 Caddy **2.11.4 Windows amd64** 二进制按官方 SHA-512 校验后，用实际 `Caddyfile` 和隔离 HTTP fixture 完成 **19 项检查**：首页静态文件、同域 API 路径、管理登录、匿名/伪造/编码路径访问拒绝、管理员密钥仅在认证后注入、API 404、NDJSON 流式响应和 smoke 脚本。首个流式事件约 **31 ms** 到达，fixture 终态延迟 800 ms，确认没有等到回复结束才一起送达。
- 打包脚本为所有包内文件生成 SHA-256 清单，并重开 ZIP 验证 CRC 和逐文件内容一致；另外生成整个 ZIP 的 `.sha256` 校验文件。明确排除本地密钥、数据库、上传文件、模型、虚拟环境、node_modules、构建缓存和 Git 历史。

## 仍由服务器完成

- 本机 Docker 引擎未运行，因此**没有执行 Linux Docker 镜像构建、Linux 依赖安装或容器启动验收**。本地使用的是现有 Windows Python/Node 依赖，不能把它当作 Linux 依赖解析通过。Dockerfile 包含 `pip check`，在服务器构建时执行。
- 没有用本次包下载/导出 Linux 检索模型、运行真实 OpenSearch、请求模型 API、处理真实图片或验收真实浏览器会话；本地代理验收使用 fixture。
- 域名、TLS、云安全组、防火墙、服务器资源容量、真实业务路径以及容器重启后的数据持久化，都需由服务器按 `DEPLOY.md` 验收。

构建有已有的大体积 JS chunk 提示；Python 离线测试有依赖弃用提示。它们没有导致本次测试失败，不等于已完成性能或安全审计。
