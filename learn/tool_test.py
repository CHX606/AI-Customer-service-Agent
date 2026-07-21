import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)


def get_weather(city):
    return {
        "city": city,
        "weather": "晴天",
        "temperature": 25
    }


question = input("你：")


response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[
        {
            "role": "system",
            "content": """
请判断用户想做什么。

只能返回 JSON，格式如下：

{
    "action": "chat 或 get_weather",
    "city": "城市名称，没有则为 null"
}
"""
        },
        {
            "role": "user",
            "content": question
        }
    ],
    response_format={
        "type": "json_object"
    }
)


result = response.choices[0].message.content

data = json.loads(result)


if data["action"] == "get_weather":
    weather = get_weather(data["city"])

    final_response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[
            {
                "role": "system",
                "content": "请根据工具返回的数据，自然地回答用户的问题，不要编造工具中没有的信息。"
            },
            {
                "role": "user",
                "content": f"""
用户的问题：
{question}

工具返回的数据：
{weather}

请回答用户。
"""
            }
        ]
    )

    answer = final_response.choices[0].message.content

    print("AI：", answer)
    
else:
    print("这是普通聊天")