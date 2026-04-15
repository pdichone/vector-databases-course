import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# conversation = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of France?"},
#     ],
# )

# print(conversation.choices[0].message.content)


response = client.embeddings.create(
    input="Your text string goes here", model="text-embedding-3-small"
)

print(response)
print(len(response.data[0].embedding))
