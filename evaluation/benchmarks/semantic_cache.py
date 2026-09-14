"""使用真实中文 Embedding 模型校准语义问答缓存阈值。"""

from back.knowledge.retrieval.cache import _cosine_similarity
from back.knowledge.retrieval.embeddings import get_embedding_model


POSITIVE_PAIRS = [
    ("退款政策是什么？", "请问怎样申请退款？"),
    ("套餐要怎么续费？", "如何给现有套餐续费？"),
    ("续费以后流量为什么没重置？", "套餐续费后流量没有恢复怎么办？"),
    ("客户端在哪里下载？", "怎么下载你们的软件？"),
    ("订阅链接怎么导入？", "如何把订阅地址添加到客户端？"),
    ("节点连接不上怎么办？", "无法连接服务器节点该怎么处理？"),
    ("你们的营业时间是什么？", "客服几点上班？"),
    ("支持哪些付款方式？", "我可以用什么方式支付？"),
]

NEGATIVE_PAIRS = [
    ("退款政策是什么？", "客户端在哪里下载？"),
    ("套餐要怎么续费？", "如何申请退款？"),
    ("流量什么时候重置？", "账号密码忘了怎么办？"),
    ("订阅链接怎么导入？", "支持哪些付款方式？"),
    ("节点连接不上怎么办？", "你们的营业时间是什么？"),
    ("怎么下载客户端？", "套餐价格是多少？"),
    ("如何注册账号？", "退款多久到账？"),
    ("客服联系方式是什么？", "续费后流量会增加吗？"),
]

HARD_NEGATIVE_PAIRS = [
    ("退款需要满足什么条件？", "退款申请后多久到账？"),
    ("套餐要怎么续费？", "续费后流量什么时候重置？"),
    ("节点连接不上怎么办？", "节点连接后为什么无法上网？"),
    ("订阅链接怎么导入？", "订阅链接怎么更新？"),
    ("套餐价格是多少？", "套餐到期以后怎么续费？"),
    ("账号应该如何注册？", "账号密码忘记了怎么办？"),
    ("如何下载客户端？", "客户端安装失败怎么办？"),
    ("你们的营业时间是什么？", "怎么联系人工客服？"),
]


def main() -> None:
    model = get_embedding_model()
    unique_questions = sorted(
        {
            question
            for pair in [*POSITIVE_PAIRS, *NEGATIVE_PAIRS, *HARD_NEGATIVE_PAIRS]
            for question in pair
        }
    )
    vectors = {question: model.embed_query(question) for question in unique_questions}

    positive_scores = [
        _cosine_similarity(vectors[left], vectors[right])
        for left, right in POSITIVE_PAIRS
    ]
    negative_scores = [
        _cosine_similarity(vectors[left], vectors[right])
        for left, right in NEGATIVE_PAIRS
    ]
    hard_negative_scores = [
        _cosine_similarity(vectors[left], vectors[right])
        for left, right in HARD_NEGATIVE_PAIRS
    ]

    print("正例相似度：", ", ".join(f"{score:.4f}" for score in positive_scores))
    print("负例相似度：", ", ".join(f"{score:.4f}" for score in negative_scores))
    print(
        "困难负例相似度：",
        ", ".join(f"{score:.4f}" for score in hard_negative_scores),
    )
    print(
        "范围：",
        f"正例 min={min(positive_scores):.4f} max={max(positive_scores):.4f};",
        f"负例 min={min(negative_scores):.4f} max={max(negative_scores):.4f};",
        f"困难负例 min={min(hard_negative_scores):.4f} "
        f"max={max(hard_negative_scores):.4f}",
    )
    for threshold in (0.86, 0.87, 0.88, 0.90, 0.94):
        true_positive = sum(score >= threshold for score in positive_scores)
        false_positive = sum(score >= threshold for score in negative_scores)
        hard_false_positive = sum(
            score >= threshold for score in hard_negative_scores
        )
        print(
            f"阈值 {threshold:.2f}: 正例命中 {true_positive}/{len(positive_scores)}, "
            f"普通负例误命中 {false_positive}/{len(negative_scores)}, "
            f"困难负例误命中 {hard_false_positive}/{len(hard_negative_scores)}"
        )


if __name__ == "__main__":
    main()
