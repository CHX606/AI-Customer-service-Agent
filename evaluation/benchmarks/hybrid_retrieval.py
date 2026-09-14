"""使用真实本地知识库评估压缩后的混合检索与重排质量。"""

from time import perf_counter

from evaluation.benchmarks.cases import RETRIEVAL_CASES
from back.knowledge.retrieval.hybrid import RRF_CANDIDATES, retrieve_documents_multi_query
from back.knowledge.retrieval.reranker import rerank_documents


def run_evaluation() -> tuple[int, int, float]:
    clear_cases = [case for case in RETRIEVAL_CASES if case["type"] == "clear"]
    hits = 0
    started = perf_counter()

    for case in clear_cases:
        candidates = retrieve_documents_multi_query(
            [case["question"]],
            tenant_id="default",
            limit=RRF_CANDIDATES,
        )
        results = rerank_documents(case["question"], candidates)
        sections = [str(doc.metadata.get("section_id", "")) for doc in results]
        matched = any(section in case["expected_sections"] for section in sections)
        hits += int(matched)
        print(
            f"{case['id']} {'PASS' if matched else 'FAIL'} "
            f"expected={case['expected_sections']} actual={sections}"
        )

    elapsed = perf_counter() - started
    print(f"Hit@5={hits}/{len(clear_cases)} elapsed={elapsed:.3f}s")
    return hits, len(clear_cases), elapsed


if __name__ == "__main__":
    hit_count, total, _ = run_evaluation()
    raise SystemExit(0 if hit_count == total else 1)
