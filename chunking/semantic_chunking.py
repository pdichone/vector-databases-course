# Install dependencies
# pip install langchain langchain-openai langchain-experimental chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Sample document with distinct topics
document = """
# Authentication Guide

## OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer <token>'.

## Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

## Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

## Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
"""


recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", ". ", " "]
)

recursive_chunks = recursive_splitter.split_text(document)


def detect_topics(chunk):
    """Detect which section headers appear in a chunk."""
    topics = []
    for topic in [
        "OAuth2 Authentication",
        "Rate Limiting",
        "Error Handling",
        "Webhooks",
    ]:
        if topic.lower() in chunk.lower():
            topics.append(topic)
    return topics


def print_chunks(chunks, method_name):
    """Print chunks with topic detection and clear formatting."""
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks):
        topics = detect_topics(chunk)
        topic_label = f" | Topics: {', '.join(topics)}" if topics else ""
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars){topic_label} ---")
        # Show first 3 lines for a quick preview
        lines = chunk.strip().split("\n")
        preview = "\n".join(lines[:3])
        if len(lines) > 3:
            preview += f"\n  ... ({len(lines) - 3} more lines)"
        print(preview)


print_chunks(recursive_chunks, "RECURSIVE CHUNKING (fixed-size, 400 chars)")


semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,  # Split at 90th percentile dissimilarity
)

semantic_chunks = semantic_chunker.split_text(document)


# Create two vector stores - one for each chunking method
recursive_vectorstore = Chroma.from_texts(
    recursive_chunks, embeddings, collection_name="recursive_chunks"
)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks, embeddings, collection_name="semantic_chunks"
)


# Test queries
test_queries = [
    "How do I authenticate with OAuth2?",
    "What happens when I hit the rate limit?",
    "How are webhooks secured?",
    "What format are errors returned in?",
]


def test_retrieval(query, vectorstore, name):
    results = vectorstore.similarity_search(query, k=1)
    print(f'\\n{name} - Query: "{query}"')
    print(f"Retrieved: {results[0].page_content[:150]}...")
    return results[0].page_content


print(f"\n{'='*60}")
print("  RETRIEVAL TESTS")
print(f"{'='*60}")

for query in test_queries:
    print("=" * 60)
    recursive_result = test_retrieval(query, recursive_vectorstore, "RECURSIVE")
    semantic_result = test_retrieval(query, semantic_vectorstore, "SEMANTIC")

# print_chunks(semantic_chunks, "SEMANTIC CHUNKING (meaning-based splits)")

# # Comparison summary
# print(f"\n{'='*60}")
# print("  COMPARISON SUMMARY")
# print(f"{'='*60}")
# print(f"  {'Method':<30} {'Chunks':<10} {'Avg Size':<10} {'Topic Mixing'}")
# print(f"  {'-'*60}")

# for name, chunks in [
#     ("Recursive (fixed-size)", recursive_chunks),
#     ("Semantic (meaning-based)", semantic_chunks),
# ]:
#     avg_size = sum(len(c) for c in chunks) / len(chunks)
#     # Count chunks that contain multiple topics
#     mixed = sum(1 for c in chunks if len(detect_topics(c)) > 1)
#     mix_label = f"{mixed}/{len(chunks)} chunks" if mixed else "None"
#     print(f"  {name:<30} {len(chunks):<10} {avg_size:<10.0f} {mix_label}")

# print(f"\n  Key Insight: Recursive splitting uses fixed character limits,")
# print(f"  so it may split mid-topic or merge unrelated topics.")
# print(f"  Semantic splitting uses embedding similarity to find natural")
# print(f"  topic boundaries, keeping related content together.\n")
