from back.evaluation.retrieval_cases import (
    RETRIEVAL_CASES,
)
from back.rag.vectorstore import get_vector_store


TOP_K = 5


def retrieve_unique_sections(
    vector_store,
    question: str,
    limit: int = TOP_K,
):
    """检索并按照章节编号去重。"""

    raw_results = (
        vector_store.similarity_search_with_score(
            query=question,
            k=limit * 3,
        )
    )

    unique_results = []
    seen_section_ids = set()

    for document, distance in raw_results:
        section_id = str(
            document.metadata.get(
                "section_id",
                "",
            )
        )

        if section_id in seen_section_ids:
            continue

        seen_section_ids.add(section_id)

        unique_results.append(
            (
                document,
                float(distance),
            )
        )

        if len(unique_results) >= limit:
            break

    return unique_results


def find_expected_rank(
    results,
    expected_sections: list[str],
):
    """查找正确章节首次出现的排名。"""

    expected_section_set = set(
        expected_sections
    )

    for rank, (document, _) in enumerate(
        results,
        start=1,
    ):
        section_id = str(
            document.metadata.get(
                "section_id",
                "",
            )
        )

        if section_id in expected_section_set:
            return rank

    return None


def show_results(results):
    """打印检索到的章节和距离。"""

    for rank, (document, distance) in enumerate(
        results,
        start=1,
    ):
        section_id = document.metadata.get(
            "section_id",
            "未知",
        )

        section_title = document.metadata.get(
            "section_title",
            "未知",
        )

        print(
            f"  第 {rank} 名："
            f"{section_id} {section_title}，"
            f"距离={distance:.4f}"
        )


def evaluate_clear_cases(vector_store):
    """评测信息明确、具有正确章节的问题。"""

    clear_cases = [
        case
        for case in RETRIEVAL_CASES
        if case["type"] == "clear"
    ]

    hit_counts = {
        1: 0,
        3: 0,
        5: 0,
    }

    reciprocal_rank_sum = 0.0

    print("\n" + "=" * 70)
    print("开始评测明确问题")
    print("问题数量：", len(clear_cases))

    for case in clear_cases:
        question = case["question"]
        expected_sections = case[
            "expected_sections"
        ]

        results = retrieve_unique_sections(
            vector_store=vector_store,
            question=question,
            limit=TOP_K,
        )

        expected_rank = find_expected_rank(
            results=results,
            expected_sections=(
                expected_sections
            ),
        )

        print("\n" + "-" * 70)
        print("ID：", case["id"])
        print("问题：", question)
        print(
            "期望章节：",
            expected_sections,
        )

        show_results(results)

        if expected_rank is None:
            print("结果：FAIL，正确章节未进入Top 5")
        else:
            print(
                "结果：PASS，正确章节排名：",
                expected_rank,
            )

            reciprocal_rank_sum += (
                1 / expected_rank
            )

            for k in hit_counts:
                if expected_rank <= k:
                    hit_counts[k] += 1

    total = len(clear_cases)

    print("\n" + "=" * 70)
    print("向量检索评测结果")

    for k, hit_count in hit_counts.items():
        hit_rate = (
            hit_count / total
            if total
            else 0
        )

        print(
            f"Hit@{k}："
            f"{hit_count}/{total} "
            f"({hit_rate:.2%})"
        )

    mrr = (
        reciprocal_rank_sum / total
        if total
        else 0
    )

    print(f"MRR：{mrr:.4f}")


def inspect_non_clear_cases(vector_store):
    """观察模糊问题和知识库外问题的距离分布。"""

    non_clear_cases = [
        case
        for case in RETRIEVAL_CASES
        if case["type"] != "clear"
    ]

    print("\n" + "=" * 70)
    print("观察模糊和知识库外问题")

    for case in non_clear_cases:
        results = retrieve_unique_sections(
            vector_store=vector_store,
            question=case["question"],
            limit=TOP_K,
        )

        print("\n" + "-" * 70)
        print("ID：", case["id"])
        print("类型：", case["type"])
        print("问题：", case["question"])
        print(
            "期望动作：",
            case["expected_action"],
        )

        show_results(results)


def main():
    """运行向量检索离线评测。"""

    vector_store = get_vector_store()

    evaluate_clear_cases(
        vector_store
    )

    inspect_non_clear_cases(
        vector_store
    )


if __name__ == "__main__":
    main()