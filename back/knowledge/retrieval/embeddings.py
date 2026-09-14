"""用户提出的问题向量化模块。"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from functools import lru_cache



MODEL_NAME = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model():
    """创建并缓存中文 Embedding 模型。"""

    from langchain_huggingface import HuggingFaceEmbeddings

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
    vector = embedding_model.embed_query("如何添加一个新的客服账号？")
    print("向量长度：", len(vector))
    print("向量前 10 个数字：", vector[:10])