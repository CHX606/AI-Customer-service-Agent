"""
工具根据用户问题，从已经建立索引的知识库中检索相关 chunk，把这些 chunk 整理成参考资料返回给 Agent，最后由 Agent 生成回答。
"""

from langchain_core.tools import tool

from back.knowledge.retrieval.service import retrieve_documents

@tool
def search_knowledge_base(question:str) -> str:
    """查询可乐云客服知识库。

    当用户询问账号、套餐、续费、流量、节点、软件使用、
    订单或其他可乐云客服问题时，使用这个工具查找相关资料。
    """

    documents = retrieve_documents(question)

    if not documents:
        return "知识库中没有找到相关资料。"
    
    results = []

    for index, document in enumerate(documents, start=1):
        result = (
            f"参考资料 {index}：\n"
            f"{document.page_content}"
        )

        results.append(result)

    return "\n\n".join(results)

if __name__ == "__main__":
    question = "续费之后为什么流量没有重置？"

    result = search_knowledge_base.invoke(
        {
            "question": question,
        }
    )

    print("测试问题：", question)
    print("\n工具返回结果：")
    print(result)