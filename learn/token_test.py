import tiktoken

text = "你好，我正在学习AI Agent"

encoding = tiktoken.get_encoding("cl100k_base")

tokens = encoding.encode(text)

print("tokens")

print("Token数量:",len(tokens))

