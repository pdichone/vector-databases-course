"""
Lecture 5: LangChain pgvector Setup
Demonstrates creating a PGVector vectorstore
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Connection string
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres"
)

# Collection name (like a table name)
COLLECTION_NAME = "documents"


def create_vectorstore():
    """Create and return a PGVector vectorstore"""

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create vectorstore
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,  # Better metadata support
    )

    return vectorstore


def main():
    print("Creating PGVector vectorstore...")

    vectorstore = create_vectorstore()
    print("✅ Vectorstore created!")

    # Verify connection by checking collection
    print(f"\n📦 Collection: {COLLECTION_NAME}")
    print(f"🔗 Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")

    print("\n✅ Setup complete! Ready to add documents.")


if __name__ == "__main__":
    main()
