"""压测单进程 CPU Reranker 的并发上限与排队开销。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

from langchain_core.documents import Document

from back.knowledge.retrieval.reranker import get_reranker_model, rerank_documents
from evaluation.benchmarks.performance import latency_summary


QUERIES = [
    "Windows 上连接成功但网页打不开怎么办？",
    "订单付款成功后套餐为什么没有到账？",
    "续费后有效期增加了但流量没有重置",
    "怎么登录 X（Twitter）？",
]

DOCUMENT_TEXTS = [
    "Windows 客户端显示已连接但无法打开网页时，先检查系统代理和虚拟网卡模式。",
    "付款成功后如套餐未到账，请刷新订阅；仍未到账时提供订单号联系客服核查。",
    "续费只延长有效期，剩余流量是否重置取决于套餐的流量重置规则。",
    "访问 X（Twitter）前需确认网络连接正常，然后使用邮箱或用户名在官网登录。",
    "节点无法连接时，可以更新订阅并切换其他可用节点后重试。",
    "忘记密码时，可在登录页面使用绑定邮箱发起密码重置。",
    "不同设备同时使用的数量取决于所购买套餐的设备限制。",
    "退款申请需要满足服务条款中的退款条件并提交订单信息。",
]


def _documents() -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={
                "section_id": f"benchmark-{index}",
                "section_title": "性能测试",
                "rrf_score": 1 / (60 + index),
            },
        )
        for index, text in enumerate(DOCUMENT_TEXTS, start=1)
    ]


def _run_one(index: int) -> dict[str, float]:
    started = perf_counter()
    result = rerank_documents(QUERIES[index % len(QUERIES)], _documents())
    elapsed_ms = (perf_counter() - started) * 1000
    metadata = result[0].metadata
    return {
        "elapsed_ms": elapsed_ms,
        "queue_wait_ms": float(metadata["reranker_queue_wait_ms"]),
        "inference_ms": float(metadata["reranker_inference_ms"]),
    }


def benchmark(limit: int, requests: int) -> dict:
    os.environ["RERANKER_MAX_CONCURRENCY"] = str(limit)
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=requests) as executor:
        futures = [executor.submit(_run_one, index) for index in range(requests)]
        samples = [future.result() for future in as_completed(futures)]
    wall_ms = (perf_counter() - started) * 1000
    return {
        "max_concurrency": limit,
        "requests": requests,
        "wall_ms": round(wall_ms, 2),
        "throughput_rps": round(requests / (wall_ms / 1000), 3),
        "elapsed_ms": latency_summary([item["elapsed_ms"] for item in samples]),
        "queue_wait_ms": latency_summary([item["queue_wait_ms"] for item in samples]),
        "inference_ms": latency_summary([item["inference_ms"] for item in samples]),
        "mean_cpu_work_ms": round(mean(item["inference_ms"] for item in samples), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limits", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # 首次模型加载与一次推理不计入并发结果。
    get_reranker_model()
    os.environ["RERANKER_MAX_CONCURRENCY"] = "1"
    _run_one(0)

    results = [benchmark(limit, args.requests) for limit in args.limits]
    payload = {"results": results}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
