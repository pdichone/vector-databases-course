import os
from dotenv import load_dotenv

load_dotenv()


from langchain.chat_models import init_chat_model


# OLD WAY (still works, but less flexible)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-4o-mini")

model = init_chat_model("gpt-4o-mini")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = model.invoke(messages)

# print(response.content)

for chunk in model.stream("explain the theory of relativity"):
    print(chunk.content, end="", flush=True)


# model = init_chat_model(
#     model="gpt-4o-mini",
#     model_provider="openai",
#     temperature=0.7,
#     max_tokens=1000
# )
