# Linux 服务器全新部署

本包包含当前源码、内置知识文档、生产构建配置和运维脚本。不包含本机 `.env`、数据库、历史会话、上传文件、依赖目录、模型权重或 Git 历史。服务器需要联网下载镜像、依赖及模型；这不是离线镜像包。历史说明可能已过时，部署以本文和当前源码为准。

## 1. 先检查服务器

- 本方案针对 Linux x86_64/amd64，CPU 必须支持 AVX2；ARM 或 Windows 主机先做适配评估。
- 建议至少 4 核、16 GB 内存，预留约 40 GB 磁盘（包含 Docker 数据目录）。这是容量建议，实际占用和并发能力要实测；ONNX 首次导出可能比常驻推理占用更多内存。
- Docker Engine、Compose V2（需支持 `up --wait`）和主机 Python 3。应用的 Python 3.12 与 Node 24 都封装在镜像内，无需装到主机。
- 检查已有容器、80/443/8080/8443 端口、反向代理和网站。保留已有业务；不要覆盖现有代理配置或卸载服务。
- 能访问 Docker Hub、Python/npm 包源、PyTorch CPU 源、Hugging Face 及选用的模型 API。不能下载时先排查网络，按需使用可信镜像源；不要静默升级依赖。

解压后进入项目目录，运行只读检查：

```bash
sha256sum -c SHA256SUMS
bash infra/deploy/preflight.sh
```

如果 `vm.max_map_count` 太小，在确认主机情况后，把 `vm.max_map_count=262144` 写入独立的 `/etc/sysctl.d/` 配置并加载；已有更大值保持不变。不要为了部署此项目关闭全机防火墙或停掉其他服务。

## 2. 创建服务器配置

```bash
python3 infra/deploy/configure.py
```

它通过官方 Caddy 镜像生成密码哈希，在当前项目创建权限为 0600 的 `.env.production` 与 `deployment-secrets.txt`，不会覆盖已有文件或打印生成的密码。首次运行可能需要拉取 `caddy:2.11.4-alpine`。

在 `.env.production` 填入 `API_KEY`、`BASE_URL`、`MODEL`。模型必须支持当前聊天接口；图片语义使用 `IMAGE_UNDERSTANDING_MODEL`，留空会沿用主模型。`RESPONSE_REASONING_EFFORT=low` 应按提供商能力调整，不支持时置空。这些参数都在服务器配置，不进入网页代码。含 `$` 的值用单引号保护；生成的 bcrypt 哈希必须保留单引号。

生成的管理账户在 `deployment-secrets.txt`。管理员使用 `https://你的域名/manage` 登录，然后点击页面右上角的管理按钮。Caddy 校验用户名/密码后，把服务器内的 `ADMIN_API_KEY` 传给后端；浏览器无需持有后端管理员密钥。不要设置 `ADMIN_AUTH_DISABLED=1`。

默认配置仅监听主机 `127.0.0.1:8080`，供验收使用。对公网开放前按第 4 节配置 HTTPS。

## 3. 构建、准备模型、初始化知识库

以下命令在项目根目录执行。辅助函数只简化 Compose 参数，不读取 shell 脚本形式的密钥文件：

```bash
dc() { docker compose --env-file .env.production -f compose.deploy.yaml "$@"; }
dc config --quiet
dc build --pull backend web
dc run --rm --no-deps backend python infra/deploy/prepare_models.py
dc up -d --wait --wait-timeout 600 opensearch
dc run --rm --no-deps backend python -m scripts.migrate_to_opensearch --tenant default
dc up -d --wait --wait-timeout 900
python3 infra/deploy/smoke.py http://127.0.0.1:8080
```

`config --quiet` 只做校验，避免把展开的密钥打印到日志。模型准备在同一个持久卷中完成，包括中文 Embedding、AVX2 INT8 Reranker 和 PaddleOCR-VL；脚本会检查向量维度、精排推理和 OCR 加载。首次下载、导出与知识库构建可能较慢。失败就修复具体原因再重试，不要跳过模型检查后宣称全部功能可用。

新库会创建项目默认的可乐云企业资料，并把包内 `resources/knowledge/` 注册为知识源。这些是源码内置内容；没有迁移本机用户数据。迁移脚本只应在首次初始化或明确需要重建时执行，它会重建该租户的 OpenSearch 文档。

## 4. 域名和 HTTPS

如果服务器没有现有网站代理，域名已指向此服务器，可将 `.env.production` 改为：

```dotenv
SITE_ADDRESS=chat.example.com
HTTP_BIND=0.0.0.0:80
HTTPS_BIND=0.0.0.0:443
```

把示例替换为真实域名，在云安全组及主机防火墙允许 TCP 80/443，执行 `dc up -d web`。Caddy 会申请和续期证书。不要把示例域名当作真实配置。

如果已有 Nginx/Caddy/宝塔代理，保持本包 `SITE_ADDRESS=:80`，选择空闲的回环端口（例如 `HTTP_BIND=127.0.0.1:18080`），由现有 HTTPS 虚拟主机代理到该端口。外层代理应保留请求路径、Host 和 Authorization，允许至少 25 MB 请求体、至少 300 秒读取超时并关闭响应缓冲，确保图片上传、管理登录和流式回答正常。

后端 8000、OpenSearch 9200/9600 均不映射到公网。OpenSearch 使用单节点容器的默认证书，仅在专用内部网络内连接，因此客户端关闭该证书校验；跨主机或外部数据库必须换用可信证书和正式凭据。公网入口始终使用 HTTPS。

暂时没有域名时，可先通过 SSH 隧道访问回环端口验收，再确定正式 HTTPS 入口。上线前结合现有云网关/代理设置适合业务的访问限流；目前默认租户聊天允许公众访问，会消耗模型 API 额度。

## 5. 验收：不能只看容器处于 running

```bash
dc ps
python3 infra/deploy/smoke.py https://真实域名 --admin --chat
dc exec -T backend python -c 'from back.infrastructure.search.opensearch import get_opensearch_client,get_vector_store; c=get_opensearch_client(); print(c.cluster.health()["status"]); n=get_vector_store("default").count(); print("indexed chunks:",n); assert n>0'
```

`--admin` 交互读取管理密码；`--chat` 会调用真实聊天链路并写入一条带 `deploy-smoke-` 前缀的验收会话。脚本检查首页、同域 API、启动健康、匿名管理访问 401、管理登录和流式回答终态。`/health` 只反映启动检查，不代表模型 API、当前 OpenSearch 连接与全部业务都正常。

还应实际验证：

1. 浏览器打开首页并进行文本问答，流式内容逐步出现；核对一个有知识库依据的问题，不能仅凭非空回复判定模型成功。
2. `/manage` 登录后查看/修改企业资料，上传一份临时测试知识文件并确认索引成功；检查图片问答能处理一张测试截图。清理自己创建的测试文件。
3. 执行 `dc restart backend opensearch` 后等待健康，再检查企业资料、知识源列表、文档数和同一个验收会话仍存在。网页对话列表存放于浏览器本地，服务端会话状态在 `sessions.db`。
4. 检查公网只开放需要的入口，记录域名、安装目录、镜像 ID/digest、模型准备结果和资源占用。代码验证与服务器真实验收分别记录。

## 6. 持久化、备份、更新

默认 Compose 项目名固定为 `ai-customer-service`，避免换目录后意外创建空卷。同一服务器部署第二套时应明确修改项目名及镜像标签，不与本实例混用。

| Docker 卷 | 内容 |
| --- | --- |
| `ai-customer-service_app-data` | SQLite 业务/会话库、上传文件、OCR 数据、模型及缓存 |
| `ai-customer-service_search-data` | OpenSearch 持久索引 |
| `ai-customer-service_caddy-data` | 证书及 ACME 状态 |
| `ai-customer-service_caddy-config` | Caddy 运行配置目录 |

`dc down` 保留卷；不要执行 `down -v`、`docker volume prune` 或删除数据目录。生产更新前先保存旧版本源码、镜像 ID/标签和配置副本，再备份上述卷及 `.env.production`。需要应用卷的一致快照时短暂停止本项目的 web/backend；冷备 OpenSearch 时也要先停止本项目的 OpenSearch，或者使用官方 snapshot。不能直接把正在写入的数据库文件当作可靠备份。备份文件与密钥权限保持为 0600，备份结束后恢复原来运行的服务。

更新使用相同项目名和卷，重新构建后 `dc up -d --wait` 并重复验收。回滚使用保存的旧代码/镜像与同一份卷；若曾做数据格式迁移，则按备份恢复，不盲目套用旧镜像。不要自动重新执行知识库全量重建。

查看日志：`dc logs --tail=100 backend opensearch web`。日志中可能有业务内容，分享前脱敏。容器采用 `unless-stopped` 重启策略及日志轮转；主机须启用 Docker 开机启动。

## 官方参考

- [Compose 启动依赖与健康检查](https://docs.docker.com/compose/how-tos/startup-order/)
- [OpenSearch Docker 与系统参数](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/)
- [Caddy 反向代理与流式响应](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy)
- [Caddy 密码认证](https://caddyserver.com/docs/caddyfile/directives/basic_auth)
- [PaddleOCR 依赖安装](https://www.paddleocr.ai/main/en/version3.x/installation.html)
