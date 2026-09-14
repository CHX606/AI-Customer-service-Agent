# 客服系统性能测试

测试集共覆盖五类问题：现有知识库原句、口语化改写、信息不足追问、X/Twitter 等外网常用应用、越权/无关问题。`coverage_gap` 表示当前知识库还没有，但生产版最终应该能回答；这类失败会单独记为 `knowledge_gap_or_scope_routing`。

## 1. 先校验测试集

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --suite full --dry-run
```

## 2. 单独测 OpenSearch 检索和 Reranker

先确保 OpenSearch 已启动、知识库已重建：

```powershell
docker compose --env-file infra\opensearch\.env.local -f infra\opensearch\compose.yml up -d
.\.venv\Scripts\python.exe -m scripts.migrate_to_opensearch --tenant default
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --repeats 3 --enforce
```

只比较模型实力时保持候选数和融合权重不变，分别在独立进程运行：

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --repeats 1 --reranker-model BAAI/bge-reranker-base --output evaluation/results/reranker-base.json
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --repeats 1 --reranker-model BAAI/bge-reranker-v2-m3 --output evaluation/results/reranker-v2-m3.json
```

这里统计纯检索 P50/P95、重排 P50/P95、Hit@1/3/5、MRR。只有带 `expected_sections` 的现有知识库用例参与检索准确率计算。

### ONNX CPU 优化

模型只需导出一次；产物保存在 `data/models/`，不会提交到 Git：

```powershell
.\.venv\Scripts\python.exe -m scripts.export_reranker_onnx --variant all --quantization avx512
```

分别测试 O3 FP32、INT8 十二候选和 INT8 八候选：

```powershell
$modelPath="data/models/bge-reranker-v2-m3-onnx"
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --reranker-backend onnx --reranker-model-path $modelPath --reranker-onnx-file onnx/model_O3.onnx --reranker-candidates 12 --output evaluation/results/reranker-v2-m3-onnx-o3-c12.json
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --reranker-backend onnx --reranker-model-path $modelPath --reranker-onnx-file onnx/model_qint8_avx512.onnx --reranker-candidates 12 --output evaluation/results/reranker-v2-m3-onnx-int8-c12.json
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode retrieval --suite full --repeats 3 --reranker-backend onnx --reranker-model-path $modelPath --reranker-onnx-file onnx/model_qint8_avx512.onnx --reranker-candidates 8 --output evaluation/results/reranker-v2-m3-onnx-int8-c8-production.json --enforce
```

当前 Ryzen 7 7840H 的三轮验收结果：重排 P50 1003.46ms、P95 1084.83ms、Hit@5 91.11%、MRR 0.8063。若需要立即回退，只需设置 `RERANKER_BACKEND=torch`；PyTorch 后端会自动忽略 ONNX 路径配置。

## 3. 端到端测试

测试基线建议关闭语义缓存，避免缓存命中掩盖真实检索耗时：

```powershell
$env:PUBLIC_CHAT_TENANTS="default"
$env:SEMANTIC_CACHE_ENABLED="0"
.\.venv\Scripts\python.exe -m uvicorn back.api:app --host 127.0.0.1 --port 8000
```

生产请求默认启用 Router V2：明确单轮问题走确定性快速通道，其他单轮问题走最小结构化路由；存在历史消息或活动问题时继续使用完整上下文分析器，保留指代消解、问题纠正、追问回答和查询改写。可分别通过 `ROUTER_MODEL`、`CONTEXT_MODEL` 和 `RESPONSE_MODEL` 配置路由、复杂多轮分析和最终回答模型；异常会自动回退，紧急回滚只需设置 `ROUTER_V2_ENABLED=0`。

当前生产配置中，复杂多轮分析使用 `gpt-5.3-codex-spark`，最终回答使用 `gpt-5.6-luna`。24 条请求分析测试均正确，复杂多轮 12/12 正确；相较 `gpt-5.4-mini`，复杂多轮 P95 从 11081.21ms 降至 4585.17ms。14 条回答 smoke 测试质量均为 85.71%，Luna 将回答 P95 从 11168.85ms 降至 10123.30ms。

最终回答进一步采用 `RESPONSE_REASONING_EFFORT=low`，压缩重复规则和结构字段，但保留全部 5 份精排证据正文。完整 63 条测试结果如下：

| 并发 | 质量通过率 | 首 Token P95 | 完整答案 P95 | 回答生成 P95 | 接口错误率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 80.95% | 6577.88ms | 7120.73ms | 6454.89ms | 0% |
| 5 | 80.95% | 7206.44ms | 9369.31ms | 6673.40ms | 0% |
| 10 | 80.95% | 10697.23ms | 11614.85ms | 6740.69ms | 0% |

相较 Router V2 初始版本，完整答案 P95 在并发 1/5/10 下分别从 14300.35/14692.15/18632.32ms 降至 7120.73/9369.31/11614.85ms。三档并发质量一致且略有提升，因此保留当前配置。

CPU Reranker 使用进程内有界信号量限制同时推理数，超出的请求进入队列。可通过 `RERANKER_MAX_CONCURRENCY` 和 `RERANKER_QUEUE_TIMEOUT_SECONDS` 调整。当前机器 10 请求突发压测中，并发上限 1/2/4 的吞吐分别为 5.70/7.39/8.30 次/秒，重排完成 P95 分别为 1671.03/1341.01/1193.12ms，因此默认设为 4。

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmarks.reranker_concurrency --limits 1 2 4 --requests 10
```

另开终端运行：

```powershell
# 快速冒烟
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode http --suite smoke --concurrency 1

# 完整并发测试
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode http --suite full --concurrency 5 --repeats 2 --enforce

# 只测 X/Twitter
.\.venv\Scripts\python.exe -m evaluation.benchmarks.performance --mode http --suite full --category external_x_twitter
```

如果租户需要 Token，不要写进测试集，使用环境变量：

```powershell
$env:BENCHMARK_TENANT_TOKEN="你的租户令牌"
```

报告会自动写入 `evaluation/results/`，同时生成 JSON 原始数据和 Markdown 汇总。

### 追问闭环测试

单轮测试只能验证第一次回复。追问闭环测试会为每条用例复用同一个
`session_id`，先发送信息不足的问题，再发送用户补充信息，分别统计首轮合理
追问率、补充信息后解决率、上下文保留率和完整闭环通过率：

```powershell
.\.venv\Scripts\python.exe -m evaluation.benchmarks.multiturn --dry-run
.\.venv\Scripts\python.exe -m evaluation.benchmarks.multiturn --concurrency 1 --output evaluation/results/multiturn-current.json
.\.venv\Scripts\python.exe -m evaluation.benchmarks.multiturn --case clarify_site_to_chatgpt_blank --output evaluation/results/multiturn-chatgpt-blank.json
```

`context_retention_rate` 专门检查系统是否重复询问用户已经提供的设备、软件、
平台或故障范围。`final_resolution_rate` 只看补充信息后的最终回复，首轮追问本身
不会被当作已经解决。

## 4. 指标含义

- `first_event_ms`：用户多久看到第一个处理状态。
- `first_token_ms`：用户多久看到回答正文的第一个 Token；知识库回答只有在证据状态和支持资料编号通过校验后才开放输出。
- `time_to_answer_ms`：最终完整答案返回耗时。
- `routing_ms`：从“理解问题”到“查询知识库”，主要看请求分析与路由。
- `route_source`：记录请求命中的路由层，包含 `fast_rule`、`light_model`、`legacy_context`、`legacy_full`、`fallback_full` 和 `semantic_cache`。
- `retrieval_stage_ms`：从“查询知识库”到“筛选相关资料”，主要看 Embedding 与 OpenSearch。
- `rerank_stage_ms`：CrossEncoder 重排耗时。
- `reranker_queue_wait_ms`：等待 CPU Reranker 并发名额的耗时。
- `reranker_inference_ms`：真正执行 ONNX 推理的耗时。
- `generation_stage_ms`：证据判断和最终回答生成耗时。
- `knowledge_gap_or_scope_routing`：外网应用问题被当成无关问题，或知识库没有对应资料。
- `answer_content`：已经回答，但缺少测试要求的关键内容。

默认生产门槛写在测试集顶部，可以按部署机器调整。首次优化时优先看失败类型：检索慢调 OpenSearch/Embedding，重排慢调 Reranker，生成慢调模型调用链；外网问题答不出先补知识库和业务范围，积累足够的真实失败样本后再考虑模型微调。
