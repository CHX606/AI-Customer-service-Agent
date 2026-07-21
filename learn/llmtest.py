import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL")
)

messages=[
    {
        "role": "system",
        "content": "你是一名专业的Python老师，请用简单易懂的方式回答问题。"
    }
]

while True:
    question = input("你：")
    if question == "退出":
        break
    
    messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages = messages,
        stream = True
    )

    print("AI: ",end="")

    answer = ""

    for chunk in stream:
        content = chunk.choices[0].delta.content

        if(content):
            print(content,end="",flush=True)
            answer+=content

    print()

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    if(len(messages) > 20):
        messages = messages[-20:]