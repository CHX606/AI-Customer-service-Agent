import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# 加载.env
load_dotenv()


# 创建客户端
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


# =====================
# 1. 定义真正执行的工具
# =====================

def get_weather(city):
    return {
        "city": city,
        "weather": "晴天",
        "temperature": 25
    }


# =====================
# 2. 告诉模型有哪些工具
# =====================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "需要查询天气的城市，例如广州"
                    }
                },
                "required": [
                    "city"
                ]
            }
        }
    }
]


# =====================
# 3. 用户输入
# =====================

question = input("你：")


messages = [
    {
        "role": "user",
        "content": question
    }
]


# =====================
# 4. 第一次调用LLM
# 判断是否需要工具
# =====================

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=messages,
    tools=tools
)


message = response.choices[0].message


# =====================
# 5. 如果模型调用工具
# =====================

if message.tool_calls:

    tool_call = message.tool_calls[0]


    print(
        "调用工具:",
        tool_call.function.name
    )


    print(
        "参数:",
        tool_call.function.arguments
    )


    # JSON字符串转Python字典
    arguments = json.loads(
        tool_call.function.arguments
    )


    # =====================
    # 6. Python执行工具
    # =====================

    if tool_call.function.name == "get_weather":

        weather = get_weather(
            arguments["city"]
        )


        print(
            "工具结果:",
            weather
        )


        # =====================
        # 7. 保存assistant工具调用消息
        # =====================

        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                ]
            }
        )


        # =====================
        # 8. 保存工具返回结果
        # =====================

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    weather,
                    ensure_ascii=False
                )
            }
        )


        # =====================
        # 9. 第二次调用LLM
        # 生成最终回答
        # =====================

        final_response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages
        )


        answer = final_response.choices[0].message.content


        print(
            "AI:",
            answer
        )


else:

    # 不需要工具，直接回答

    print(
        "AI:",
        message.content
    )