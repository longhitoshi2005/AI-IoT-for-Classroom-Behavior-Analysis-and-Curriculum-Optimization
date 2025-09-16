from ai_client import chat

messages = [
    {"role": "user", "content": "Xin chào, bạn có thể giúp tôi không?"}
]

response = chat(messages)
print("Bot:", response)