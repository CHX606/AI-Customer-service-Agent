import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

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

print("完整数据：", data)
print("操作：", data["action"])
print("城市：", data["city"])