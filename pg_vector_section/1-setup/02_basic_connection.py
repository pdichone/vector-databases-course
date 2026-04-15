"""
Lecture 2: Basic PostgreSQL Connection
Demonstrates connecting to PostgreSQL and enabling pgvector
"""

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# Connection settings
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect():
    """Connect to PostgreSQL and return connection"""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    return conn


def enable_pgvector(conn):
    """Enable the pgvector extension"""
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✅ pgvector extension enabled")
    cur.close()


def verify_pgvector(conn):
    """Verify pgvector is working"""
    cur = conn.cursor()

    # Check extension
    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    result = cur.fetchone()

    if result:
        print(f"✅ pgvector version: {result[4]}")
    else:
        print("❌ pgvector not installed")

    # Test vector type
    cur.execute("SELECT '[1,2,3]'::vector;")
    vector = cur.fetchone()[0]
    print(f"✅ Vector test: {vector}")

    cur.close()


def main():
    print("Connecting to PostgreSQL...")
    conn = connect()
    print("✅ Connected!")

    print("\nEnabling pgvector...")
    enable_pgvector(conn)

    print("\nVerifying installation...")
    verify_pgvector(conn)

    conn.close()
    print("\n✅ Setup complete!")


if __name__ == "__main__":
    main()
