"""
用户提出的问题，向量化，然后与数据库中已有的chunk向量进行相似度计算，返回最相似的chunk作为候选结果。
"""
from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def get_embedding_model():
    """创建并返回中文 Embedding 模型。"""

    embedding_model = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "prompt": "为这个句子生成表示以用于检索相关文章：",
            "normalize_embeddings": True,
        },
    )

    return embedding_model


if __name__ == "__main__":
    embedding_model = get_embedding_model()

    vector = embedding_model.embed_query(
        "如何添加一个新的客服账号？"
    )

    print("向量长度：", len(vector))
    print("向量前 10 个数字：", vector[:10])