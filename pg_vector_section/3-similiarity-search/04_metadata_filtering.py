"""
Lecture 3: Metadata Filtering with pgvector
Demonstrates combining vector search with SQL WHERE clauses
"""

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def get_embedding(text: str) -> list[float]:
    """Generate embedding for query text"""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def search_with_category_filter(conn, query: str, category: str, limit: int = 5):
    """Search within a specific category"""
    query_embedding = get_embedding(query)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            content,
            1 - (embedding <=> %s::vector) AS similarity,
            metadata
        FROM documents
        WHERE metadata->>'category' = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, category, query_embedding, limit),
    )

    results = cur.fetchall()
    cur.close()
    return results


def search_with_multiple_filters(
    conn,
    query: str,
    category: str = None,
    topic: str = None,
    min_similarity: float = 0.5,
    limit: int = 5,
):
    """Search with multiple optional filters"""
    query_embedding = get_embedding(query)

    # Build dynamic query
    conditions = ["1=1"]  # Always true base condition
    params = [query_embedding]

    if category:
        conditions.append("metadata->>'category' = %s")
        params.append(category)

    if topic:
        conditions.append("metadata->>'topic' = %s")
        params.append(topic)

    where_clause = " AND ".join(conditions)

    # Add remaining params
    params.extend([query_embedding, min_similarity, query_embedding, limit])

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            content,
            1 - (embedding <=> %s::vector) AS similarity,
            metadata
        FROM documents
        WHERE {where_clause}
          AND 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        params,
    )

    results = cur.fetchall()
    cur.close()
    return results


def search_recent_documents(conn, query: str, days: int = 7, limit: int = 5):
    """Search only recent documents"""
    query_embedding = get_embedding(query)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            content,
            1 - (embedding <=> %s::vector) AS similarity,
            metadata,
            created_at
        FROM documents
        WHERE created_at > NOW() - INTERVAL '%s days'
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, days, query_embedding, limit),
    )

    results = cur.fetchall()
    cur.close()
    return results


def print_results(results, title: str):
    """Pretty print search results"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print("=" * 60)

    if not results:
        print("No results found")
        return

    for i, result in enumerate(results, 1):
        content = result[0]
        score = result[1]
        metadata = result[2]
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Content: {content[:70]}...")
        print(f"   Metadata: {metadata}")


def main():
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connected to PostgreSQL")

    query = "How do I make vector search faster?"

    # Search without filter
    print(f"\n🎯 QUERY: '{query}'")

    # Filter by category
    results = search_with_category_filter(conn, query, category="database", limit=3)
    print_results(results, "Category = 'database'")

    # Filter by category = 'ml'
    results = search_with_category_filter(conn, query, category="ml", limit=3)
    print_results(results, "Category = 'ml'")

    # Multiple filters
    results = search_with_multiple_filters(
        conn, query, category="database", min_similarity=0.3, limit=3
    )
    print_results(results, "Category='database' + min_similarity=0.3")

    # Recent documents only
    results = search_recent_documents(conn, query, days=30, limit=3)
    print_results(results, "Last 30 days only")

    conn.close()


if __name__ == "__main__":
    main()
