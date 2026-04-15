"""
Lecture 3: Insert Vectors into pgvector
Demonstrates generating embeddings and storing in PostgreSQL
"""

import os
import psycopg2
from psycopg2.extras import execute_values, Json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
client = OpenAI()  # Uses OPENAI_API_KEY from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate embedding for text using OpenAI"""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def setup_table(conn):
    """Create the documents table if it doesn't exist"""
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(1536),
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT NOW()
        );
    """
    )

    conn.commit()
    cur.close()
    print("✅ Table ready")


def insert_document(conn, content: str, metadata: dict = None):
    """Insert a single document with its embedding"""
    embedding = get_embedding(content)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (content, embedding, metadata)
        VALUES (%s, %s, %s)
        RETURNING id
        """,
        (content, embedding, Json(metadata or {})),
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()

    return doc_id


def insert_documents_batch(conn, documents: list[dict]):
    """Insert multiple documents efficiently"""
    # Generate embeddings for all documents
    contents = [doc["content"] for doc in documents]

    print(f"Generating embeddings for {len(contents)} documents...")
    response = client.embeddings.create(model="text-embedding-3-small", input=contents)

    # Prepare data for batch insert
    data = []
    for i, doc in enumerate(documents):
        data.append(
            (doc["content"], response.data[i].embedding, Json(doc.get("metadata", {})))
        )

    # Batch insert
    cur = conn.cursor()
    execute_values(
        cur,
        """
        INSERT INTO documents (content, embedding, metadata)
        VALUES %s
        """,
        data,
        template="(%s, %s, %s)",
    )
    conn.commit()
    cur.close()

    print(f"✅ Inserted {len(documents)} documents")


def main():
    # Connect to PostgreSQL
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connected to PostgreSQL")

    # Setup table
    setup_table(conn)

    # Sample documents
    documents = [
        {
            "content": "PostgreSQL is a powerful open-source relational database with advanced features.",
            "metadata": {"category": "database", "topic": "postgresql"},
        },
        {
            "content": "pgvector is a PostgreSQL extension for vector similarity search.",
            "metadata": {"category": "database", "topic": "pgvector"},
        },
        {
            "content": "Vector embeddings capture semantic meaning of text as numerical representations.",
            "metadata": {"category": "ml", "topic": "embeddings"},
        },
        {
            "content": "HNSW indexes provide fast approximate nearest neighbor search.",
            "metadata": {"category": "database", "topic": "indexes"},
        },
        {
            "content": "RAG combines retrieval with language model generation for accurate responses.",
            "metadata": {"category": "ml", "topic": "rag"},
        },
        {
            "content": "LangChain provides abstractions for building LLM-powered applications.",
            "metadata": {"category": "ml", "topic": "langchain"},
        },
        {
            "content": "Cosine similarity measures the angle between two vectors in high-dimensional space.",
            "metadata": {"category": "math", "topic": "similarity"},
        },
        {
            "content": "Production deployments should use connection pooling for efficiency.",
            "metadata": {"category": "devops", "topic": "production"},
        },
    ]

    # Insert documents
    insert_documents_batch(conn, documents)

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents;")
    count = cur.fetchone()[0]
    print(f"\n📊 Total documents in database: {count}")
    cur.close()

    conn.close()


if __name__ == "__main__":
    main()
