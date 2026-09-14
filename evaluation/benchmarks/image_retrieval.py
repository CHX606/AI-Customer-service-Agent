"""使用真实向量库评估第 7～11 张图片诊断资料的召回质量。"""

from time import perf_counter

from back.knowledge.retrieval.hybrid import RRF_CANDIDATES, retrieve_documents_multi_query
from back.knowledge.retrieval.reranker import rerank_documents


IMAGE_RETRIEVAL_CASES = [
    {
        "id": "img007",
        "question": "Shadowrocket 提示不能获取订阅节点怎么办？",
        "expected_image_order": 7,
    },
    {
        "id": "img008",
        "question": "公告里的备用订阅地址在哪里？",
        "expected_image_order": 8,
    },
    {
        "id": "img009",
        "question": "Clash Verge 导入订阅时出现 failed to fetch 和 TLS verifier 错误",
        "expected_image_order": 9,
    },
    {
        "id": "img010",
        "question": "Windows 手动代理的 127.0.0.1 和 10808 应该怎么设置？",
        "expected_image_order": 10,
    },
    {
        "id": "img011",
        "question": "Mac App Store 商店菜单里怎么退出登录账户？",
        "expected_image_order": 11,
    },
]


def run_evaluation() -> tuple[int, int, float]:
    hits = 0
    started = perf_counter()
    for case in IMAGE_RETRIEVAL_CASES:
        question = str(case["question"])
        candidates = retrieve_documents_multi_query(
            [question],
            tenant_id="default",
            limit=RRF_CANDIDATES,
        )
        results = rerank_documents(question, candidates)
        image_orders = [
            int(document.metadata["image_order"])
            for document in results
            if document.metadata.get("content_type") == "image_diagnostic"
        ]
        expected = int(case["expected_image_order"])
        matched = expected in image_orders
        hits += int(matched)
        print(
            f"{case['id']} {'PASS' if matched else 'FAIL'} "
            f"expected={expected} actual={image_orders}"
        )

    elapsed = perf_counter() - started
    print(f"Image Hit@5={hits}/{len(IMAGE_RETRIEVAL_CASES)} elapsed={elapsed:.3f}s")
    return hits, len(IMAGE_RETRIEVAL_CASES), elapsed


if __name__ == "__main__":
    hit_count, total, _ = run_evaluation()
    raise SystemExit(0 if hit_count == total else 1)
