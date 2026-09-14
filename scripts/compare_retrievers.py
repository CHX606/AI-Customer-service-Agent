"""
该文件为测试BM25与BGE的检索结果对比，三类问题，由两个模型内置的分数或举例来排名，
作用为：
1，哪种问题更适合BM25或向量检索。
2，正确chunk是否进入Top 5。
3，哪些问题容易召回错误内容。
4，为后续混合检索和RRF融合提供依据。
"""

from back.knowledge.retrieval.hybrid import (
    retrieve_keyword_candidates,
    retrieve_vector_candidates,
)
from back.knowledge.retrieval.text import tokenize_chinese


TEST_QUESTIONS = [
    {
        "type": "精确关键词",
        "question": "续费之后为什么流量没有重置？",
    },
    {
        "type": "口语化改写",
        "question": "我刚充完钱，怎么可用额度还是没变化？",
    },
    {
        "type": "知识库无答案",
        "question": "北京明天会下雨吗？",
    },
]


def format_text(text: str) -> str:
    """删除多余换行，方便查看检索结果。"""

    return " ".join(text.split())


def show_bm25_results(
    results,
    limit: int = 5,
):
    """打印 BM25 检索结果及分数。"""

    print("\nBM25 关键词检索：")

    if not results:
        print("没有检索结果")
        return

    for index, (document, score) in enumerate(
        results[:limit],
        start=1,
    ):
        text = format_text(
            document.page_content
        )

        print(
            f"\n第 {index} 名，"
            f"BM25 分数：{score:.4f}"
        )
        print(text[:300])


def show_vector_results(
    results,
    limit: int = 5,
):
    """打印向量检索结果及距离。"""

    print("\nBGE 向量检索：")

    if not results:
        print("没有检索结果")
        return

    for index, (document, distance) in enumerate(
        results[:limit],
        start=1,
    ):
        text = format_text(
            document.page_content
        )

        print(
            f"\n第 {index} 名，"
            f"向量距离：{distance:.4f}"
        )
        print(text[:300])


def main():
    """对比关键词检索和向量检索的原始排名。"""

    for test_case in TEST_QUESTIONS:
        question_type = test_case["type"]
        question = test_case["question"]

        print("\n" + "=" * 70)
        print("问题类型：", question_type)
        print("用户问题：", question)
        print(
            "BM25 分词：",
            tokenize_chinese(question),
        )

        bm25_results = (
            retrieve_keyword_candidates(question, limit=5)
        )

        vector_results = (
            retrieve_vector_candidates(question, limit=5)
        )

        show_bm25_results(
            bm25_results,
        )

        show_vector_results(
            vector_results,
        )


if __name__ == "__main__":
    main()
