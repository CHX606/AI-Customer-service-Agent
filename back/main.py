from langchain_core.messages import HumanMessage

from back.agent.graph import customer_service_graph


EXIT_COMMANDS = {
    "退出",
    "exit",
    "quit",
    "q",
}


def main():
    """启动命令行客服程序。"""

    print("可乐云 AI 客服已启动")
    print("输入“退出”可以结束对话")

    messages = []

    while True:
        try:
            question = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n对话已结束")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("AI 客服：感谢使用，再见！")
            break

        request_messages = [
            *messages,
            HumanMessage(content=question),
        ]

        try:
            result = customer_service_graph.invoke(
                {
                    "messages": request_messages,
                }
            )
        except Exception as error:
            print("AI 客服：本次请求失败，请稍后重试。")
            print("错误信息：", error)
            continue

        messages = result["messages"]

        final_answer = messages[-1].content

        print("\nAI 客服：", final_answer)


if __name__ == "__main__":
    main()
