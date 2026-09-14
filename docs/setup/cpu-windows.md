# AI Customer Service Agent：另一台电脑环境搭建（CPU 版）

更新时间：2026-08-06

适用场景：

- Windows 10/11 64 位。
- 另一台电脑没有 NVIDIA 显卡。
- 后端、RAG、PaddleOCR-VL 均使用 CPU。
- 项目代码通过 Git 或其他方式复制到新电脑。

> CPU 电脑不要安装 `paddlepaddle-gpu`，也不需要安装 CUDA、cuDNN 或 `nvcc`。
>
> PaddleOCR-VL 可以使用 CPU，但解析速度会明显慢于 GPU。这不影响继续开发和功能验证。

## 1. 建议硬件与基础软件

建议配置：

- 64 位 x86 CPU，并支持 AVX 指令集。
- 16 GB 或更多内存。
- 至少预留 10 GB 磁盘空间，用于虚拟环境和三个模型。

安装以下软件：

```text
Python 3.12.x 64-bit（当前已验证版本：3.12.10）
Node.js 24.x（当前已验证版本：24.18.0）
npm 11.x（当前已验证版本：11.16.0）
Git
VS Code
```

验证：

```powershell
py -3.12 --version
node --version
npm --version
git --version
```

## 2. 获取项目

在新电脑克隆或复制项目代码。

不需要复制以下内容：

```text
.venv/
front/node_modules/
C:\Users\<用户名>\.cache\huggingface\
```

进入项目根目录。以下命令假设终端当前目录就是：

```text
AI Customer service Agent
```

## 3. 创建项目虚拟环境

```powershell
py -3.12 -m venv .venv

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
```

后续命令始终明确使用项目中的：

```text
.venv\Scripts\python.exe
```

这样不会把依赖安装到系统 Python。

## 4. 安装 CPU 版 PyTorch

当前项目的 BGE Embedding 和 Reranker 使用 PyTorch。原电脑使用的是 CPU 版：

```text
torch 2.13.0+cpu
```

安装：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  torch==2.13.0 `
  --index-url https://download.pytorch.org/whl/cpu
```

## 5. 安装 CPU 版 PaddlePaddle

不要安装 `paddlepaddle-gpu`。

安装与原电脑同系列的 CPU 版本：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  paddlepaddle==3.2.2 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

验证 CPU Paddle：

```powershell
& ".\.venv\Scripts\python.exe" `
  -c "import paddle; print('Paddle:', paddle.__version__); print('CUDA:', paddle.is_compiled_with_cuda()); paddle.utils.run_check()"
```

CPU 电脑的正常结果应包含：

```text
Paddle: 3.2.2
CUDA: False
PaddlePaddle is installed successfully!
```

## 6. 安装项目后端依赖

执行：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  "fastapi==0.140.13" `
  "uvicorn[standard]==0.51.0" `
  "python-dotenv==1.2.2" `
  "langchain==1.3.14" `
  "langchain-core==1.5.1" `
  "langchain-community==0.4.2" `
  "langchain-openai==1.3.5" `
  "langchain-huggingface==1.2.2" `
  "langchain-text-splitters==1.1.2" `
  "langgraph==1.2.9" `
  "opensearch-py==3.2.0" `
  "docx2txt==0.9" `
  "jieba==0.42.1" `
  "sentence-transformers==5.6.1" `
  "transformers==5.14.1" `
  "safetensors==0.8.0" `
  "paddleocr[doc-parser]==3.7.0" `
  "paddlex==3.7.2"
```

安装结束后，固定经过验证的冲突解决版本：

```powershell
& ".\.venv\Scripts\python.exe" -m pip install `
  "numpy==2.3.5" `
  "PyYAML==6.0.2" `
  "kubernetes==35.0.0"
```

原因：

```text
paddlex 3.7.2 需要 PyYAML == 6.0.2
较新的 kubernetes 36.0.3 需要 PyYAML >= 6.0.3
paddlex 的文档解析组件会使用 kubernetes 客户端
```

因此项目固定使用 `kubernetes 35.0.0`。

检查所有依赖：

```powershell
& ".\.venv\Scripts\python.exe" -m pip check
```

预期：

```text
No broken requirements found.
```

## 7. 配置后端环境变量

在项目根目录创建 `.env`：

```env
API_KEY=填写真实的接口密钥
BASE_URL=填写接口地址
MODEL=填写使用的 LLM 名称
```

注意：

- 不要把真实密钥发送到聊天、公开仓库或截图中。
- `.env` 应加入项目根目录的 `.gitignore`。
- 当前仓库的 `.env` 已被 Git 跟踪，后续需要取消跟踪并补充 `.env.example`。

## 8. 下载三个模型

执行一次：

```powershell
& ".\.venv\Scripts\python.exe" `
  -c "from huggingface_hub import snapshot_download; models=['BAAI/bge-small-zh-v1.5','BAAI/bge-reranker-base','PaddlePaddle/PaddleOCR-VL-1.6']; [print(m, '=>', snapshot_download(repo_id=m)) for m in models]"
```

默认模型缓存位置：

```text
C:\Users\<新电脑用户名>\.cache\huggingface\hub\
├── models--BAAI--bge-small-zh-v1.5
├── models--BAAI--bge-reranker-base
└── models--PaddlePaddle--PaddleOCR-VL-1.6
```

模型缓存、虚拟环境和源代码应分开管理，不要把模型提交到 Git。

## 9. 验证 PaddleOCR-VL 的 CPU 加载

先指定 PaddleX 辅助缓存目录：

```powershell
$env:PADDLE_PDX_CACHE_HOME = Join-Path $env:USERPROFILE ".cache\paddlex"
```

再执行加载测试：

```powershell
& ".\.venv\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; from paddleocr import PaddleOCRVL; path=snapshot_download(repo_id='PaddlePaddle/PaddleOCR-VL-1.6', local_files_only=True); PaddleOCRVL(pipeline_version='v1.6', vl_rec_model_dir=path, device='cpu', use_layout_detection=False); print('PaddleOCR-VL CPU 模型加载成功')"
```

首次加载会比较慢。出现下面内容即可：

```text
PaddleOCR-VL CPU 模型加载成功
```

## 10. 图片模型的迁移说明

当前 `back/knowledge/images/parser.py` 已通过模型 ID 动态解析本机缓存，不再依赖某台电脑的绝对路径。部署新机器时需要先安装 CPU 版 PaddlePaddle，并预下载模型：

```python
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
snapshot_download(repo_id=MODEL_ID, local_files_only=True)
```

运行时约束：

```text
通过 PaddlePaddle/PaddleOCR-VL-1.6 模型 ID 解析缓存路径
device="cpu"
模型缺失时立即报错，不在服务启动阶段自动下载
```

图片功能代码位于 `back/knowledge/images/`，DOCX 图片提取位于 `back/knowledge/ingestion/docx/`。

## 11. 安装前端依赖

进入前端目录：

```powershell
Set-Location front
npm ci
npm run dev
```

项目已有 `package-lock.json`，因此使用 `npm ci`，不要复制原电脑的 `node_modules`。

## 12. 启动后端

回到项目根目录：

```powershell
Set-Location ..

& ".\.venv\Scripts\python.exe" -m uvicorn back.api:app --reload
```

## 13. OpenSearch 数据库

当前运行链路不再使用 Chroma。SQLite 保存租户与知识源元数据，OpenSearch
持久化 BM25 关键词索引、BGE 向量和 RRF 融合结果。

先创建仅本机使用的 OpenSearch 配置：

```powershell
Copy-Item infra\opensearch\.env.example infra\opensearch\.env.local
```

将 `infra\opensearch\.env.local` 中的
`OPENSEARCH_INITIAL_ADMIN_PASSWORD` 修改为符合复杂度要求的密码，并确保它与
项目 `.env` 中的 `OPENSEARCH_PASSWORD` 一致。然后启动数据库并重建索引：

```powershell
docker compose --env-file infra\opensearch\.env.local `
  -f infra\opensearch\compose.yml up -d

& ".\.venv\Scripts\python.exe" -m scripts.migrate_to_opensearch --tenant default
```

验证容器和索引：

```powershell
docker compose --env-file infra\opensearch\.env.local `
  -f infra\opensearch\compose.yml ps

& ".\.venv\Scripts\python.exe" -m evaluation.benchmarks.performance `
  --mode retrieval --suite full --repeats 1 --enforce
```

完整链路为：

```text
知识源登记（SQLite）
→ 结构化切块
→ BGE Embedding
→ OpenSearch BM25 + HNSW 向量召回
→ OpenSearch RRF 融合
→ ONNX INT8 Reranker 精排
```

`data/chroma_db` 是迁移前遗留目录，当前程序不会读取；完成备份后可另行归档。

## 14. 最终检查清单

```text
[ ] Python 3.12 64 位
[ ] 项目 .venv 已创建
[ ] CPU PyTorch 已安装
[ ] CPU PaddlePaddle 已安装
[ ] paddle.is_compiled_with_cuda() 返回 False
[ ] PaddleOCRVL 可以导入
[ ] pip check 无冲突
[ ] .env 已在本机安全配置
[ ] 三个模型已下载到 Hugging Face 缓存
[ ] 前端 npm ci 成功
[ ] 后端可以启动
[ ] OpenSearch 容器健康状态为 healthy
[ ] OpenSearch 知识索引可以重新构建
[ ] 完整检索基准通过
```

## 15. 官方参考

- PaddlePaddle Windows PIP 安装：<https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html>
- PyTorch CPU 安装选择器：<https://docs.pytorch.org/get-started/locally/>
- PaddleOCR 安装：<https://www.paddleocr.ai/main/en/version3.x/installation.html>
- PaddleOCR-VL-1.6 模型：<https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6>
