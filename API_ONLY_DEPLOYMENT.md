# 当前部署：GPT-5.5 API 识图

更新：2026-09-19 UTC，子域名迁移与 Cloudflare 橙云验收完成。本文是当前运行方案；`SERVER_DEPLOYMENT.md` 中先前的本地 OCR 尝试属于历史记录。

## 地址与凭据

- 网站：<https://ai.chx1008.com/>
- 管理入口：<https://ai.chx1008.com/manage>；浏览器密码认证后点击右上角「管理后台」。
- 管理用户名/密码：服务器文件 `/root/ai-customer-service/deployment-secrets.txt`，权限 0600；不在日志或本文展示。
- 项目目录：`/root/ai-customer-service`。
- HTTPS、文字/图片问答、管理页和同时重启后的持久性验收均已通过。管理页图片处理说明已修正、重新构建发布并经真实浏览器复验。
- 2026-09-19 按用户要求从根域名迁移到 `ai.chx1008.com` 并开启 Cloudflare 代理。本次未重建/重启应用容器、未修改用户名/密码、未迁移或删除业务数据。浏览器可能需要在新域名重新认证；旧域名的浏览器本地聊天历史不自动跨域迁移，不能将此视为后端数据丢失。

## 已授权的运行模式

用户明确同意以已有 GPT-5.5 API 替代本地 Paddle OCR，而不是关闭全部图片功能。

```text
资料中的图片 + 相邻正文 → GPT-5.5 结构化语义卡 → 缓存 → 分块/向量/关键词索引
客户截图 → 已知图匹配或 GPT-5.5 理解 → 检索现有知识 → 有依据的客服回答
```

- `IMAGE_FEATURES_ENABLED=1`、`IMAGE_SEMANTIC_ENABLED=1`、`LOCAL_OCR_ENABLED=0`、`PRELOAD_OCR_MODEL=0`。
- 图片 API 报错、熔断、低置信度时不启动本地 OCR，提示重试或改用文字。
- 资料图片预处理要求全部完成；低质量/失败结果不会静默标记为建库成功。已完成缓存可复用。
- 客户截图只用于当前问答，不自动变成长期知识。服务端保留会话文字，不长期保存上传原图。
- 两份内置文档已生成 88 个分块：Word 文档 76 个（包含全部 11 张图片语义）、应用指南 12 个。
- DOCX 上传/重索支持上述图片语义入库；PDF 当前只提取文字，不支持扫描件或其中图片的语义建库。客户聊天截图与管理员知识文档上传是两个不同入口。
- 本地保留中文 Embedding 和 AVX2 INT8 Reranker。服务器不需要 GPU，但检索/精排仍使用 CPU；图片和回答 API 会产生上游用量。

## HTTPS 与现有服务隔离

- 子域名 `ai.chx1008.com` 的 A 记录源站为 `67.215.235.111`，Cloudflare 已代理（橙云）；公网 DNS 返回 Cloudflare 节点 IP。
- 用户在 Cloudflare 新增仅匹配 `http.host eq "ai.chx1008.com"` 的 SSL「严格」配置规则，未要求变更其他主机名的 SSL 模式。没有读取 Cloudflare 凭据或启用 1Panel 管理 API。
- 复用现有 `1Panel-openresty-gYvn` 的 443 端口，独立配置 `/opt/1panel/www/conf.d/ai-customer-service.conf`。
- 上游为 `127.0.0.1:18080`，保留管理认证、关闭响应缓冲，允许 25 MiB 请求体和 300 秒读取时间。
- 80 端口仍属于原 AxonHub 服务，没有抢占或改造。橙云边缘已实测 HTTP→HTTPS 301，保留路径和查询参数；源站未新增 HTTP 跳转。若将来关掉橙云，应使用 HTTPS，不可假定直接访问源站 80 会进入客服。
- 只在客服虚拟主机中载入 `security/cloudflare-realip.conf`：信任 2026-09-19 官方清单中的 15 个 IPv4 和 7 个 IPv6 网段，通过 `CF-Connecting-IP` 恢复真实访客 IP。直连来源、回环或任意 `X-Forwarded-For` 不受信任，禁止改成 `0.0.0.0/0` / `::/0`。
- 登录/聊天限流均使用恢复后的地址。向上游重写 `X-Real-IP`、`X-Forwarded-For` 和 `CF-Connecting-IP`，去掉未经验证的 `Forwarded`；访问日志记录访客及原连接节点，不记录认证头、Cookie 或查询参数。
- API 与管理响应使用 `Cache-Control: no-store`。不要另设忽略此设置的「缓存所有内容」规则来缓存 `/api/*` 或 `/manage`。保留流式响应的无缓冲配置及 300 秒源站超时；Cloudflare 自身仍有独立限制。
- 后端 8000、OpenSearch 9200/9600 均未映射公网。
- 聊天入口每 IP 6 次/分钟、突发 3 次、每 IP 同时 2 个请求、全站同时 3 个请求；超限返回 429。静态资源、健康检查、管理 API 不受这组聊天限额影响。
- `qujian-1panel` 按用户此前要求保持停止，数据和镜像未删；其他五个既有服务保留运行。
- 本站虚拟主机是独立配置文件，未直接改写 1Panel 的网站数据库，也未开启 1Panel 管理 API。
- 本站日志单独由 `/etc/logrotate.d/ai-customer-service` 轮转（每日检查、10 MiB 阈值、保留 7 份）；规则只读调试通过，系统 logrotate 定时器已启用。

## 登录防爆破与安全巡检（2026-09-19 补齐）

- 用户批准登录防爆破和原巡检更新，未批准接入新数据备份；不添加后台 IP 白名单，不修改 SSH、防火墙或备份脚本。
- `/manage` 和 `/api/admin` 全部子路径共享来源 IP 失败计数：600 秒固定窗口内 10 次带凭据的上游 401 后，该 IP 的后台访问暂停 900 秒，返回 429 和 `Retry-After`。匿名认证挑战、成功访问和其他错误不计为错误密码。
- 后台另设每 IP 30 请求/分钟（突发 10）、全站 120 请求/分钟（突发 20）。正常前台聊天不受后台封禁影响，原聊天限流仍保留。
- 后台响应禁止缓存；不记录密码，仅信任来自 Cloudflare 官方网段的指定真实 IP 头，忽略直连客户端伪造值。保留既有 1Panel WAF 的 access/log 钩子，新增逻辑仅使用 rewrite/header-filter 阶段。
- 计数使用有界共享内存；普通平滑重载保留，OpenResty 完全重启后清零。存储不足时只拒绝后台带凭据的请求，不影响前台。该措施不等于 MFA，也不能保证拦住大量分散 IP 的攻击。
- 当前域名已开启 Cloudflare 代理，可信代理与真实来源地址配置已完成；实际公网验收证实源站记录的访客 IP 与 Cloudflare trace 一致，原连接 IP 属于 Cloudflare 官方网段。
- 原五分钟安全巡检正式登记三个客服容器；18080/18443 只接受准确的回环绑定，不全局放行端口；后端与搜索禁止主机端口发布，并验证内部搜索网络、容器健康、敏感文件、HTTPS、管理认证、证书和安全配置漂移。
- `qujian-1panel` 被明确登记为应停止；意外启动仍会告警。暂停其网站可用性探测，但保留证书、敏感文件和容器权限检查。原 SSH、WireGuard、1Panel MFA、防火墙、备份新鲜度和邮件告警检查未取消。
- 验证：7 个离线巡检测试、28 项 Lua 断言通过。正式 HTTPS 实测 10 次错误密码后四种后台路径（含编码/重复斜杠）均为 429；伪造来源头不能绕过；其他 IP 正确登录 200，受限 IP 的网站/健康 200、无效聊天请求 422。没有调用付费模型。
- 2026-09-19 12:21:42 UTC，`server-security-check.service` 实际运行 `Result=success`、退出码 0；五分钟 timer 仍 enabled/active。原恢复通知单元随后成功执行。
- 回滚副本：`security-change-20260919-2NEffQ/`。源文件位于 `infra/deploy/`；实际防护文件在 `/opt/1panel/www/sites/ai-customer-service/security/`，巡检入口仍为 `/usr/local/sbin/server-security-check`。

### 自动备份的明确边界

目前客服知识资料、图片语义缓存、企业资料与会话数据库位于新的持久卷，正常容器重启会保留；**这不等于已有独立备份**。原每日备份与 R2 加密异地备份仍服务于旧清单，没有自动包含客服业务数据及搜索索引。发生磁盘损坏或误删时，不能依靠原备份完整恢复新增数据。

用户本次只要求解释备份，因此未改动备份清单、未启动新的数据上传或备份任务。本地/异地备份脚本、主机防火墙和 SSH 配置的变更前后 SHA-256 一致。新站证书/代理配置本身处于原备份已覆盖的网站目录，不代表客服业务卷也被覆盖。

## 证书续签

- 使用用户在 1Panel 申请的 `ai.chx1008.com` 证书，记录 ID 6，自动续签开启，有效至 2026-12-18 12:01:13 UTC。旧根域名证书没有删除，但不再作为本站当前证书。
- 1Panel 负责 DNS-01 续签；`ai-customer-service-certificate.timer` 每小时检查已完成的证书。
- 同步器只读取这个域名的证书，不读取 Cloudflare API 凭据、不修改 1Panel 数据库。
- 先验证可信链、域名、有效期和密钥匹配，再以原子链接切换证书版本；配置校验/重载失败时恢复旧版本。私钥 0600，证书目录 0700。
- 实际链/域名校验、错误域名拒绝、模拟重载失败回退，以及 systemd 实际执行都已通过。
- 日志：`journalctl -u ai-customer-service-certificate.service --no-pager`。

## 子域名与橙云验收（2026-09-19）

- 新域名源站链、域名、有效期、密钥匹配及公网 Cloudflare HTTPS 均正常；首页和同源 JavaScript 资源 200，HTTP 跳转 HTTPS 保留路径/查询。
- 15 项离线安全/运行配置测试通过。迁移后在专用回环 IP 上重新验证：10 次错误密码后后台 429、伪造 `CF-Connecting-IP` / `X-Forwarded-For` 等不能绕过，其他 IP 和前台不受影响。
- 实际橙云链路：访客 IP 恢复正确，健康 200，匿名后台 401，正确凭据的管理页、企业资料及知识库列表 200，动态接口禁止缓存。
- 经橙云的一次真实文字问答通过：首个事件 0.126 秒、完整答案 25.864 秒，收到 `status` / `token` / `final`，没有错误事件。不是对所有请求的延迟保证。
- 本次新域名迁移未更改识图业务代码；图片能力沿用之前已通过的 API-only 验收，并未将旧域名浏览器截图误标为新域名截图。
- 证书同步与五分钟安全巡检均成功，两个 timer 仍 active/enabled；`hub` 公网实测 200，其他五个既有服务继续运行，`qujian` 保持停止。SSH、防火墙、备份脚本 SHA-256 未变化。
- 复验工具：`infra/deploy/verify_cloudflare_edge.py`；添加 `--chat` 会调用一次真实付费模型。脚本不会打印凭据或资料内容，也不跟随携带认证头的跨域重定向。
- 迁移前配置副本：`domain-change-20260919-i5emx5/`，仅 root 可读。迁移前证书链接目标为 `versions/dc2336f55416950e7135800c`；回滚域名必须同时恢复虚拟主机、证书同步器和巡检目标，不可只换证书链接。

## 验收结果

已完成：

- 后端全量离线测试：454 passed、5 skipped；跳过的是包内显式联网测试，另做下列真实验收。
- 随后的搜索启动重试修复专项：26 passed（包括 3 个新增重试测试）。
- 前端：11 passed，TypeScript/Vite 生产构建通过。生产依赖审计 0 项；全部依赖仍有 5 项构建/开发依赖提示，未进行无关升级。
- 镜像 `pip check` 通过；补齐原包漏装的 `docx2txt==0.9`。
- 两次真实结构化图片 API 检查约 10 秒；全部 11 张资料图完成并校验缓存。
- 真实文字流式问答：首事件 0.041 秒，样例完整回答 26.858 秒；正确区分续费与流量重置。
- 图片完整接口：已知图命中 `exact_sha256`、未知图使用 `vision_model`，均返回与知识库一致的处理建议；期间知识源、分块数和语义缓存未变化。样例约 73～83 秒（当时并行运行回归测试，不代表稳定性能上限）。
- 管理登录、匿名拒绝、资料修改、临时 Markdown 上传及真实索引通过。
- 内置 Word 管理重索的源文件缺失问题已修复；Word/Markdown 两个真实管理重索请求通过，分别 76/12 个分块。
- 后端与 OpenSearch 同时重启验收通过：自动等待搜索服务后恢复健康，OpenSearch 为 green；企业资料、上传文档及两条真实图片会话/答案保留。
- 曾发现上述同时重启会过早记录永久降级，已增加默认 180 秒启动重试并用真实同时重启验证修复。服务进程未加载 Paddle 库。
- 正式 HTTPS 域名上的真实 Chromium 文字问答与刷新后历史保留通过，本次完整文字回答 34.45 秒。
- 正式 HTTPS 浏览器文件选择、截图上传、识图回答通过，本次完整图片回答 57.68 秒；页面无未捕获 JavaScript 异常。
- 正式 HTTPS 管理密码认证、非空企业资料表单、两份知识文档列表通过。
- 最终前端重新构建时 11 个测试全部通过，TypeScript/Vite 构建通过；更新后管理页和 HTTPS 健康/匿名拒绝复验通过。
- 限流实测：8 个畸形 JSON 请求返回 4 次 422、4 次 429；不调用模型，随后健康接口仍 200、匿名管理仍 401。
- 已恢复原企业资料并删除仅本次验收上传的唯一 Markdown 文档；正式知识保留。未删除用户原有资料或其他业务数据。

页面截图：`deployment-browser-chat.png`、`deployment-browser-image-chat.png`、`deployment-browser-management.png`。

## 资源与边界

- 3 vCPU、3915 MiB 内存的共享服务器，采用 API 识图、单后端 worker 和单路本地精排，并设置 CPU/内存限额。
- 已移除部署阶段创建的 4 GiB 临时交换文件，保留原 `/dev/vda3` 的 2 GiB swap；最终根盘约 15 GiB 可用（75% 已用），主机可用内存约 1406 MiB，swap 已用约 1605 MiB。这是收尾时快照，不代表负载峰值。
- 临时模型准备/测试容器、独立前端构建器缓存、浏览器验收容器和临时测试图片已清理；这些均可重建，正式命名卷、知识资料、原部署 ZIP 和三个验收页面截图保留。
- 客服三个正式容器均运行，无 OOM、无自动崩溃重启记录；后端/搜索健康检查通过，网页实际 HTTP 检查通过。
- 检查时其他五个既有服务仍健康，`qujian-1panel` 按用户授权保持停止，原数据保留。
- 已验证本项目容器同时重启，不为验收中断其他业务而重启整台服务器。Docker 开机启动和客服容器 `unless-stopped` 策略已核实。
- 当前适合低并发使用；未做大规模压力测试，回答耗时受本机检索/精排和上游 API 影响。不要把单张图片 API 约 10 秒误认为整个截图问答耗时。

最终镜像标识：

```text
ai-customer-service-backend:local
sha256:e75e0640b51b66e119ccf2b2173215ec414f06708fe4b0054c9a226a9743b389
ai-customer-service-web:local
sha256:f5ad85091697bfc40be2e038d10b425e3ec889757f6dc54e0855cf4e5133583d
```

## 运维

```bash
cd /root/ai-customer-service
dc() { docker compose --env-file .env.production -f compose.deploy.yaml -f compose.low-memory.yaml "$@"; }
dc config --quiet
dc ps
dc logs --tail=100 backend opensearch web
dc up -d --no-build --wait --wait-timeout 300
```

必须带上 `compose.low-memory.yaml`；它包含 API-only 开关、CPU/内存限额。不要打印展开后的 Compose 配置，其中包含密钥。

初次空库流程（不要对正在运行的知识库重复全量重建）：

```bash
dc run --rm --no-deps -e PYTHONPATH=/app backend python infra/deploy/prepare_models.py
dc run --rm --no-deps -e PYTHONPATH=/app backend python infra/deploy/prepare_reference_images.py
dc run --rm --no-deps -e PYTHONPATH=/app backend python infra/deploy/initialize_knowledge.py
```

备份 `.env.production`、私密凭据、站点配置/同步服务及四个命名持久卷。SQLite 使用一致备份或短暂停写冷备，OpenSearch 使用 snapshot 或停机冷备；禁止 `down -v` 或全局 prune。

历史快捷构建文件 `backend-runtime-patch.Dockerfile` 引用 `:packaged` 原始镜像，但用户已于 2026-09-19 明确批准删除该镜像，并要求暂不调整更新脚本。因此当前不能直接使用这个快捷构建文件，后续更新前需另行处理其基础镜像引用；现用 `:local` 镜像和正在运行的客服不受影响。新服务器从普通 backend Dockerfile 构建。原始 ZIP 与 `SHA256SUMS` 未改写，源码修复有意区别于原包。
