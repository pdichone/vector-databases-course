import os
from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model


# OLD WAY (still works, but less flexible)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none")

# gpt-6-luna is a reasoning model: reasoning_effort="none" keeps it fast and cheap
model = init_chat_model("gpt-6-luna", reasoning_effort="none")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = model.invoke(messages)

# print(response.content)

for chunk in model.stream("explain the theory of relativity"):
    print(chunk.content, end="", flush=True)


# model = init_chat_model(
#     model="gpt-6-luna",
#     model_provider="openai",
#     reasoning_effort="none",
#     temperature=0.7,
#     max_tokens=1000
# )
