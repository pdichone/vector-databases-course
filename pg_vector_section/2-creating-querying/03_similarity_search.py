"""
Lecture 3: Similarity Search with pgvector
Demonstrates different distance operators and search patterns
"""

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)


def get_embedding(text: str) -> list[float]:
    """Generate embedding for query text"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def search_cosine(conn, query: str, limit: int = 5):
    """
    Cosine similarity search (recommended for OpenAI embeddings)
    Operator: <=>
    Lower distance = more similar
    """
    query_embedding = get_embedding(query)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            content,
            1 - (embedding <=> %s::vector) AS similarity,
            metadata
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit)
    )

    results = cur.fetchall()
    cur.close()
    return results


def search_l2(conn, query: str, limit: int = 5):
    """
    L2 (Euclidean) distance search
    Operator: <->
    Lower distance = more similar
    """
    query_embedding = get_embedding(query)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            content,
            embedding <-> %s::vector AS distance,
            metadata
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit)
    )

    results = cur.fetchall()
    cur.close()
    return results


def search_inner_product(conn, query: str, limit: int = 5):
    """
    Inner product search (negative because pgvector uses <#> for max inner product)
    Operator: <#>
    More negative = more similar (for normalized vectors)
    """
    query_embedding = get_embedding(query)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            content,
            (embedding <#> %s::vector) * -1 AS similarity,
            metadata
        FROM documents
        ORDER BY embedding <#> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, limit)
    )

    results = cur.fetchall()
    cur.close()
    return results


def print_results(results, title: str):
    """Pretty print search results"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

    for i, (content, score, metadata) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Content: {content[:80]}...")
        print(f"   Metadata: {metadata}")


def main():
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connected to PostgreSQL")

    # Test queries
    queries = [
        "How does vector similarity search work?",
        "What is the best way to deploy to production?",
        "Tell me about PostgreSQL features"
    ]

    for query in queries:
        print(f"\n\n🎯 QUERY: '{query}'")

        # Cosine similarity (recommended)
        results = search_cosine(conn, query, limit=3)
        print_results(results, "Cosine Similarity (<=>)")

    conn.close()

if __name__ == "__main__":
    main()
