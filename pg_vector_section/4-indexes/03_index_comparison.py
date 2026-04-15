"""
Lecture 4: Index Performance Comparison
Demonstrates HNSW vs IVFFlat performance differences
"""

import os
import time
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


def time_query(conn, query_embedding, index_type: str, runs: int = 5):
    """Time a query multiple times and return average"""
    cur = conn.cursor()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        cur.execute(
            """
            SELECT content, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """,
            (query_embedding, query_embedding),
        )
        _ = cur.fetchall()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    cur.close()

    avg_time = sum(times) / len(times)
    return avg_time


def check_index_info(conn):
    """Get index information"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            indexname,
            indexdef,
            pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
        FROM pg_indexes
        WHERE tablename = 'documents'
          AND indexdef LIKE '%vector%';
    """
    )
    results = cur.fetchall()
    cur.close()
    return results


def set_hnsw_params(conn, ef_search: int = 40):
    """Set HNSW search parameters"""
    cur = conn.cursor()
    cur.execute(f"SET hnsw.ef_search = {ef_search};")
    cur.close()


def set_ivfflat_params(conn, probes: int = 10):
    """Set IVFFlat search parameters"""
    cur = conn.cursor()
    cur.execute(f"SET ivfflat.probes = {probes};")
    cur.close()


def main():
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connected to PostgreSQL")

    # Check current indexes
    print("\n📊 Current Indexes:")
    print("-" * 60)
    indexes = check_index_info(conn)
    for name, definition, size in indexes:
        print(f"Name: {name}")
        print(f"Size: {size}")
        print(f"Def:  {definition[:80]}...")
        print()

    # Generate test query embedding
    query = "How does vector similarity search work?"
    print(f"\n🎯 Test Query: '{query}'")
    query_embedding = get_embedding(query)

    # Time queries with different parameters
    print("\n⏱️  Query Performance:")
    print("-" * 60)

    # Test with HNSW parameters
    for ef_search in [10, 40, 100, 200]:
        set_hnsw_params(conn, ef_search)
        avg_time = time_query(conn, query_embedding, "hnsw")
        print(f"hnsw.ef_search = {ef_search:3d} → {avg_time:.2f} ms")

    print()

    # Test with IVFFlat parameters (if that index exists)
    for probes in [1, 5, 10, 20]:
        set_ivfflat_params(conn, probes)
        avg_time = time_query(conn, query_embedding, "ivfflat")
        print(f"ivfflat.probes = {probes:3d} → {avg_time:.2f} ms")

    print("\n" + "=" * 60)
    print("📝 Notes:")
    print("- Higher ef_search/probes = more accurate but slower")
    print("- HNSW ef_search=40 is a good default")
    print("- IVFFlat probes=10 is a good default")
    print("- For production: benchmark with YOUR data")

    conn.close()


if __name__ == "__main__":
    main()
