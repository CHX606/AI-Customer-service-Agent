# 服务器部署记录

> 当前方案已按用户要求改为 GPT-5.5 API-only 识图，并迁移到 `https://ai.chx1008.com`（Cloudflare 橙云）。当前结果与运维方式请看 [API_ONLY_DEPLOYMENT.md](API_ONLY_DEPLOYMENT.md)。**下文保留早期部署/本地 OCR 排障过程，不代表当前状态。**

检查日期：2026-09-19。状态：正在部署，尚未上线。镜像构建、前端局部验收、真实模型 API、中文向量、INT8 精排推理、OCR 加载全部通过；正在生成内置参考图片的真实 OCR 索引。正式域名待用户确认。

## 本轮部署进度（进行中）

- 用户已填好模型配置，Compose 必填项校验通过；没有打印或复制密钥到本记录。
- 使用实际应用客户端完成三个真实 API 请求：默认文字回复（6.15 秒）、回答模型的 reasoning 配置（3.93 秒）、图片识别（1.32 秒），全部通过。
- 按用户明确授权，仅暂时停止 `qujian-1panel`（取件处网站）；其他五个服务继续运行且健康。其镜像、挂载目录和数据未删除。
- 删除本项目专用 BuildKit 构建器 `ai-customer-service-build` 及其专属缓存卷，释放约 5 GiB；磁盘空闲由约 13 GiB 增至 18 GiB。两个最终应用镜像、模型缓存和其他业务数据均保留。该构建缓存可重新生成。
- 新增 `/root/ai-customer-service-runtime/prepare.swap`（4 GiB）和 `quantize.swap`（2 GiB），权限均 0600；加上已有交换分区总计约 8 GiB。仅临时启用，尚未写入开机配置。
- 原始模型准备脚本在 2 GiB RAM、内存加 swap 总上限 7 GiB、1.5 CPU 下重试，成功完成 FP32 ONNX 导出；随后 INT8 量化在 09:27:26 UTC 再次触发容器 OOM。已清理该失败容器的约 2.1 GiB 重复临时文件，持久卷中的完整基础模型保留。
- 新增 `quantize_existing_reranker.py`，直接对已保存模型使用相同的 `AutoQuantizationConfig.avx2(is_static=False)`，不同时保留 CrossEncoder 的 FP32 推理会话；改用外部权重格式保存，模型结构验证成功后才发布目标文件。初版独立量化在 09:38:07 UTC 仍触发 9 GiB 总内存限额，失败容器随后移除，卷未删除。
- 第二版进一步在加载权重前计算真实 ONNX shape inference，再利用库支持的 infer metadata 复用结果，以避免基础量化器多保留一整份权重。原始节点和权重引用一致性断言通过；保留 ONNX 的文件安全校验，使用同目录普通临时图文件，不使用符号链接或硬链接。此次量化于 09:52:05 UTC 成功完成（exit 0、OOMKilled=false），最终模型结构校验通过：图文件 2,285,513 字节，外部 INT8 权重 568,789,008 字节。随后使用真实应用加载并预测，有限数值断言通过。
- INT8 图 SHA-256：`fce6b30840c372a1d61c79b89febc7c1d74f4fccc35c87da3d4241d8cab128d8`；量化配置 SHA-256：`5cbb4ebcdcbe24f990160893ee053bb693f2176cd16f730114c387fd66f53c1e`。
- PaddleOCR-VL-1.6 的官方快照 `c5630abae1d940eafe0697512a0325494b02ab42` 已下载完成（20 个文件，exit 0）。包内原始 OCR 加载流程也成功完成，库报告 checkpoint 全部权重正常加载；完整 `prepare_models.py` 输出 `All model preparation checks passed.`。该 CPU 加载过程约需 9 分钟且大量使用 swap，不能据此宣称运行性能或并存容量已合格。
- 在 INT8 真实推理通过后，只删除 `onnx/model.onnx` 和 `onnx/model.onnx_data` 两个 FP32 导出中间文件，共回收 2,271,577,901 字节；原始下载权重与已验证 INT8 模型保留，FP32 中间文件可重新生成。磁盘空闲恢复约 6.8 GiB。
- 修复镜像再次运行 `pip check` 通过，非 root 运行用户仍为 `10001:10001`。
- 将 `prepare_models.py` 改为分进程完成 FP32 导出和 INT8 量化，并给原导出脚本增加 `fp32` 选项，使该降峰方式可重复执行；不改变模型、量化参数或运行功能。已用 `backend-runtime-patch.Dockerfile` 加入部署脚本修复层和下述启动预热选项，依赖版本与业务处理逻辑未改变；原镜像保留为 `ai-customer-service-backend:packaged`。
- OpenSearch 3.8.0 已启动，2026-09-19 10:24 UTC 达到 healthy；真实集群健康为 green、单节点、3 个系统主分片、0 未分配分片。业务知识索引尚未建立。
- 新增 `compose.low-memory.yaml` 资源限制覆盖配置，解析校验通过，尚未启动业务栈：OpenSearch 512 MiB 堆/1 GiB 容器，backend 2 GiB RAM / 9 GiB 含 swap 总限额，web 128 MiB。精排 batch/concurrency 均为 1；功能和候选数量保持原值。
- 增加默认关闭的 `PRELOAD_OCR_MODEL` 启动选项，本机覆盖配置将其启用。OCR 先预热、再加载 RAG，避免客户首次触发 OCR 时等待长时间初始化，并降低并行持有权重的峰值。启动健康检查宽限 15 分钟；预热顺序、原默认行为、独立开关与错误传播测试通过。完整服务并存容量仍未验收，不保证此资源配置一定可用。
- 已在限额、无业务数据/密钥挂载的临时容器准备 Chromium / Playwright 浏览器验收环境，不修改宿主机浏览器或生产镜像依赖；待服务启动后实际执行 `verify_browser.py`。
- 源码检查发现首次知识索引会读取未随包携带的 OCR 参考图缓存；新增 `infra/deploy/prepare_reference_images.py` 调用包内既有分割/OCR 流程生成并验证这些数据，目前正在实际执行。不伪造缓存或跳过图片索引。
- 首次参考图 OCR：第 7 张原图于 10:22:10 UTC 完成并通过原程序关键词检查；接着处理第 8 张（999×995）时容器于 10:22:54 UTC OOM 退出（exit 137，OOMKilled=true），即使配置 2 GiB RAM / 9 GiB 含 swap 总上限也未完成。五个既有运行服务继续健康。第 7 张完整结果保存在持久卷，重试可复用。
- 排查发现 PaddleX 3.7.2 / PaddlePaddle 3.2.2 的 CPU Siglip attention 会完整分配图像 token 的平方矩阵。新增默认关闭、版本和源代码 AST 指纹校验的 `OCR_CPU_ATTENTION_CHUNK_SIZE` 分块选项：仅拆 query 行，各行仍访问全部 key，不缩小原图、不改变模型精度、不截断 OCR 输出。正在进行真实数值一致性与重新识别试验，尚不能称为解决。
- OpenSearch 曾在磁盘约 90% 时自动阻止创建索引。只删除本次下载、已完成 INT8 转换后不再用于运行的 `bge-reranker-v2-m3` 原始 `model.safetensors` 缓存及其快照链接（2,271,071,852 字节），不删除 INT8 模型、Embedding、OCR 权重或业务数据；可从原官方快照重新下载。空闲约 8.2 GiB，随后真实 `/_cluster/state/blocks` 返回空，未降低数据库的磁盘保护阈值。

## 来源与安装目录

- 安装目录：`/root/ai-customer-service`。
- 原始 ZIP：`/root/.codex/attachments/148d7993-79ca-427d-a9e4-f68041ca648d/ai-customer-service-server.zip`。
- ZIP SHA-256：`ffc19ab6d3dbdb8d9738d87978c5972e377138e93e672b4e70a7cd236f8efb27`。
- 解压 CRC 检查通过；首次解压后 `SHA256SUMS` 中 237 个文件全部匹配。
- 完整阅读了 `CODEX_DEPLOY.md`、`DEPLOY.md`、`DEPLOYMENT_CHECKS.md` 和部署配置/脚本。
- 原包保留，用于区分包内源码与本次服务器修复；`SHA256SUMS` 未被重写。

## 服务器与现有服务

| 项目 | 检查结果 |
| --- | --- |
| 系统 | Ubuntu 24.04.4 LTS，Linux 6.8.0-138-generic，KVM |
| CPU | 3 vCPU，Intel Xeon Gold 6152，x86_64，支持 AVX2 |
| 内存 | 3915 MiB；初始可用约 2.8 GiB；swap 2 GiB |
| 磁盘 | 根分区约 62 GiB，初始剩余约 25 GiB；Docker 数据也在此分区 |
| Docker / Compose | Docker 29.7.2，Compose v5.5.0，Docker 开机自启已启用 |
| Docker 数据目录 | `/var/lib/docker` |
| vm.max_map_count | 1048576，满足要求，无需修改 |
| 现有端口 | 80：axonhub；443/8080：OpenResty；8317、25500、13001 等已有业务 |
| 项目拟用端口 | `127.0.0.1:18080`、`127.0.0.1:18443`，配置时均空闲 |

初始运行容器为 `1Panel-openresty-gYvn`、`axonhub-postgres`、`axonhub-app`、`qujian-1panel`、`subconverter`、`cliproxyapi`。随后按用户要求暂时停止 `qujian-1panel`，其余五个保留；尚未修改既有网站或代理配置。

文档建议 4 核、16 GiB 内存和约 40 GB 空闲磁盘；这些是建议值，不是已证明的硬性最低值。本机低于该建议。首次精排导出发生容器级 OOM；正在通过专属资源限额和额外 swap 重试。完整 OCR、检索与 OpenSearch 的并存容量仍需实测。API 配置已经补齐且实际验证通过。

## 已完成检查与准备

- 包完整性、109 个原有 Python 文件语法、Bash 预检语法均通过。
- 已执行包内 `preflight.sh`；架构、AVX2、Docker 和系统参数通过，内存/磁盘触发容量提醒。
- 官方 Docker Registry、PyPI、npm、PyTorch CPU 源和三个 Hugging Face 模型仓库可访问。
- `configure.py` 已生成 `.env.production` 与 `deployment-secrets.txt`，二者权限均为 0600。密钥不写入本记录。
- 修改生成配置的回环端口为 18080/18443，避开已有网站。
- 使用独立构建器 `ai-customer-service-build`：1 CPU、1536 MiB 内存、内存加 swap 上限 2 GiB；未切换全局默认构建器。
- 前端 Linux 生产镜像 `ai-customer-service-web:local` 构建成功；元数据在 `web-build-metadata.json`。
- 后端 Linux 生产镜像 `ai-customer-service-backend:local` 构建成功，完整依赖安装与 `pip check` 通过（`No broken requirements found`）；元数据在 `backend-build-metadata.json`。
- 后端离线运行时导入检查通过：PyTorch `2.13.0+cpu`、PaddleOCRVL、CrossEncoder 和 FastAPI 应用均可导入。验证容器没有网络或业务数据挂载。
- 中文 Embedding 模型 `BAAI/bge-small-zh-v1.5` 已下载，实际中文查询返回 512 维向量，包内断言通过。
- 使用真实前端镜像进行回环 HTTP 验证：首页、构建 JS 的同域 `/api` 配置、匿名/伪造头管理访问拒绝、生成的管理账户登录均通过。临时测试容器随后自动移除。
- 前端生产依赖 `npm audit --omit=dev --package-lock-only` 报告 0 项已知漏洞。`npm ci` 对全部依赖曾报告 5 项（2 moderate、3 high），生产构建也存在大 chunk 提示。
- 模型 API 已通过本轮三个真实请求。尚未进行真实浏览器、后端业务 API、OpenSearch、上传索引、OCR 和重启持久化验收；这些不由前端或上游 API 检查替代。

## 模型准备失败的实际证据

运行的是包内原始 `infra/deploy/prepare_models.py`，未关闭 OCR、检索或鉴权，也未更换模型。测试容器限制为 1 CPU、2 GiB RAM、内存加 swap 总上限 3584 MiB，以给现有服务保留资源。

中文向量检查通过后，`scripts.export_reranker_onnx --variant int8 --quantization avx2` 在 ONNX 导出阶段被 SIGKILL。Docker 显示 `OOMKilled=true`、`exit=1`。内核在 2026-09-19 08:53:02 记录了 `Memory cgroup out of memory`；对应 cgroup 为本次容器 `9309654820de22f391c58b92b748e180793f04373e26aaf01cd57ba1b6fbbc9c`，被终止的进程为 UID 10001 的 Python。

这证明导出未能在上述受限预算内完成，不能据此将全机总内存误报为已耗尽，也不能将模型视为已准备成功。发生 OOM 后，原有 6 个服务仍健康，主机可用内存恢复到约 3 GiB。

`ai-customer-service_app-data` 持久卷中保留约 2.3 GiB 已下载模型缓存。Embedding 已通过；Reranker 原始权重已下载但 INT8 导出及推理检查未通过；OCR 下载/加载阶段尚未执行。失败的临时测试容器已清理，释放约 2.1 GiB 临时写入文件；这些临时导出内容可在资源足够后重新生成，持久模型缓存未删除。

首次尝试结束时复查：磁盘剩余约 13 GiB，主机可用内存约 3 GiB，原有 6 个服务全部健康；18080、18443、8000、9200、9600 均无监听。本项目镜像、私密配置、2.3 GiB 模型缓存保留。其后的清理、服务暂停与重试见本记录顶部。

## 本次修复

后端构建容器访问 `http://deb.debian.org` 超时。已在 `infra/deploy/backend.Dockerfile` 中将同一 Debian 官方源改用 HTTPS，并为 `apt-get update` 增加 `APT::Update::Error-Mode=any`，避免源更新失败后继续安装。重试后系统依赖安装通过。项目声明的依赖版本未更换。

新增 `infra/deploy/check_web_image.py`，用于复现前端/代理局部验收。该脚本明确不检验后端；测试期间仅监听回环地址，管理凭据通过私密配置读取，不打印到输出。

## 待补充的配置

用户已补齐 `.env.production` 的必填模型配置，默认/回答/图片客户端真实调用均成功。图片模型留空时沿用主模型，该回退已实际验证。

还需用户指定域名，或明确先使用 SSH 隧道验收。已有 HTTPS 代理为 1Panel OpenResty，新站点应在确认域名后代理到 `127.0.0.1:18080`，保留 Authorization、关闭响应缓冲、允许至少 25 MB 请求体与 300 秒读取超时。

网站地址：尚未上线。管理路径：正式站点的 `/manage`，当前未开放。管理员凭据文件：`/root/ai-customer-service/deployment-secrets.txt`。

## 后续运行、日志与备份

配置和容量就绪后，在项目目录继续执行 `DEPLOY.md` 第 3～5 节的模型准备、OpenSearch 初始化、完整启动和业务验收。辅助命令：

```bash
cd /root/ai-customer-service
dc() { docker compose --env-file .env.production -f compose.deploy.yaml -f compose.low-memory.yaml "$@"; }
dc config --quiet
dc ps
dc logs --tail=100 backend opensearch web
```

备份应覆盖 `.env.production`、私密凭据及 `DEPLOY.md` 列出的四个持久卷；使用应用停写后的冷备或 OpenSearch snapshot，保持备份权限 0600。不要将在线写入的 SQLite 文件直接当作一致备份，不执行 `down -v` 或全局清理。

当前只有本项目的 OpenSearch 服务运行，无公开网址。之后仍需完成参考图 OCR、建库、完整应用启动、HTTPS 和全部业务验收。未通过或未执行的项目不算完成。

## 实际镜像版本

| 镜像 | manifest digest |
| --- | --- |
| `ai-customer-service-web:local` | `sha256:bdd160c1b54f81bd5a7a67ca381e70cffe58b245f1b789bb616613b0287ea90d` |
| `ai-customer-service-backend:local`（部署脚本与预热修复层） | `sha256:7a4e84fb480bbf45e73dc6c6cdd34b22a4f41959c134e26bbc4a04ea685244f3` |
| `ai-customer-service-backend:packaged`（保留的原镜像） | `sha256:e6a88dd01fe3d616232d39ea361d278f38219c26e76773c247fb47231d7714ac` |
| `caddy:2.11.4-alpine` | `sha256:de23def33b17fb5d1290b0f6c2add1d70780e52341896c00a4c8a2a2fe9d355e` |
| `node:24-bookworm-slim`（构建阶段） | `sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6` |
| `python:3.12-slim-bookworm` | `sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e` |

后端 Dockerfile 修复后的 SHA-256 为 `1fe3d2429369379cf3413101c301427c857abf651fb10839eeeaa22b9e2a9894`。专用构建器及其缓存已在本轮清理，最终镜像保留。
