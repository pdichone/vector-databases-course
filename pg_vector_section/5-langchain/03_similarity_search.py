"""
Lecture 5: Similarity Search with LangChain
Demonstrates various search methods and filtering
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)
COLLECTION_NAME = "documents"


def get_vectorstore():
    """Get existing vectorstore"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def basic_search(vectorstore, query: str, k: int = 3):
    """Basic similarity search"""
    print(f"\n🔍 Basic Search: '{query}'")
    print("-" * 50)

    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:70]}...")
        print(f"   Metadata: {doc.metadata}")


def search_with_scores(vectorstore, query: str, k: int = 3):
    """Search with similarity scores"""
    print(f"\n🔍 Search with Scores: '{query}'")
    print("-" * 50)

    results = vectorstore.similarity_search_with_score(query, k=k)

    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f}")
        print(f"   {doc.page_content[:70]}...")


def search_with_filter(vectorstore, query: str, filter_dict: dict, k: int = 3):
    """Search with metadata filter"""
    print(f"\n🔍 Filtered Search: '{query}'")
    print(f"   Filter: {filter_dict}")
    print("-" * 50)

    results = vectorstore.similarity_search(query, k=k, filter=filter_dict)

    if not results:
        print("   No results found")
        return

    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:70]}...")
        print(f"   Metadata: {doc.metadata}")


def main():
    print("Connecting to vectorstore...")
    vectorstore = get_vectorstore()
    print("✅ Connected!")

    # Basic search
    basic_search(vectorstore, "How does vector search work?")

    # Search with scores
    search_with_scores(vectorstore, "What is PostgreSQL?")

    # Filtered searches
    search_with_filter(
        vectorstore, "How to improve performance?", filter_dict={"category": "database"}
    )

    search_with_filter(
        vectorstore, "How to improve performance?", filter_dict={"category": "ml"}
    )

    search_with_filter(
        vectorstore, "What is an index?", filter_dict={"topic": "indexes"}
    )

    print("\n✅ All searches complete!")


if __name__ == "__main__":
    main()
