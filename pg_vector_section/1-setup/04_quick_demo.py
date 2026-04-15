"""
Lecture 2: Quick Demo - Create table and insert a vector
Simple example to show pgvector working
"""

import psycopg2

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


def main():
    # Connect
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    print("✅ Connected!")

    # Enable pgvector extension
    print("\nEnabling pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✅ Extension enabled!")

    # Create table with vector column
    print("\nCreating items table...")
    cur.execute(
        """
        DROP TABLE IF EXISTS items;
        CREATE TABLE items (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(3)  -- 3 dimensions for demo
        );
    """
    )
    print("✅ Table created!")

    # Insert a simple vector
    print("\nInserting sample vectors...")
    cur.execute(
        """
        INSERT INTO items (content, embedding) VALUES
        ('First item', '[1.0, 2.0, 3.0]'),
        ('Second item', '[4.0, 5.0, 6.0]'),
        ('Third item', '[1.1, 2.1, 3.1]');
    """
    )
    print("✅ Vectors inserted!")

    # Query to show it works
    print("\nQuerying all items:")
    cur.execute("SELECT id, content, embedding FROM items;")
    for row in cur.fetchall():
        print(f"  ID: {row[0]}, Content: {row[1]}, Embedding: {row[2]}")

    # Similarity search demo
    print("\nSimilarity search - find items closest to [1,2,3]:")
    cur.execute(
        """
        SELECT content, embedding, embedding <-> '[1,2,3]' AS distance
        FROM items
        ORDER BY distance
        LIMIT 3;
    """
    )
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} (distance: {row[2]:.4f})")

    # Cleanup
    cur.close()
    conn.close()
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
