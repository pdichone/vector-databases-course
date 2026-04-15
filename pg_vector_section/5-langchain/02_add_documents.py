"""
Lecture 5: Add Documents to PGVector
Demonstrates adding documents with metadata
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

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


def main():
    print("Connecting to vectorstore...")
    vectorstore = get_vectorstore()

    # Sample documents with metadata
    documents = [
        Document(
            page_content="Transformers use self-attention mechanisms to process input sequences in parallel, making them highly efficient for NLP tasks.",
            metadata={"source": "research", "category": "ml", "topic": "transformers"},
        ),
        Document(
            page_content="Chunking strategies such as recursive splitting and semantic chunking directly impact retrieval quality in RAG pipelines.",
            metadata={"source": "guide", "category": "retrieval", "topic": "chunking"},
        ),
        Document(
            page_content="IVFFlat indexes partition vectors into clusters using k-means, trading some recall accuracy for significantly faster search times.",
            metadata={"source": "guide", "category": "performance", "topic": "ivfflat"},
        ),
        Document(
            page_content="Prompt engineering techniques like chain-of-thought and few-shot examples help large language models produce more accurate outputs.",
            metadata={"source": "tutorial", "category": "llm", "topic": "prompting"},
        ),
        Document(
            page_content="Hybrid search combines keyword-based BM25 scoring with dense vector similarity to improve retrieval relevance across diverse queries.",
            metadata={
                "source": "research",
                "category": "retrieval",
                "topic": "hybrid_search",
            },
        ),
        Document(
            page_content="Fine-tuning embedding models on domain-specific data can dramatically improve similarity search results compared to general-purpose models.",
            metadata={"source": "guide", "category": "ml", "topic": "fine_tuning"},
        ),
        Document(
            page_content="Metadata filtering allows narrowing vector search results by structured fields like date, category, or author before ranking by similarity.",
            metadata={"source": "docs", "category": "retrieval", "topic": "filtering"},
        ),
        Document(
            page_content="Reranking models like cross-encoders score query-document pairs jointly, providing more accurate relevance rankings than bi-encoder retrieval alone.",
            metadata={
                "source": "research",
                "category": "retrieval",
                "topic": "reranking",
            },
        ),
    ]

    print(f"Adding {len(documents)} documents...")

    # Add documents
    ids = vectorstore.add_documents(documents)

    print(f"✅ Added {len(ids)} documents")
    print(f"\nDocument IDs: {ids[:3]}...")

    # Verify by doing a quick search
    print("\n🔍 Quick verification search:")
    results = vectorstore.similarity_search("How do transformers work?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"  {i}. {doc.page_content[:60]}...")


if __name__ == "__main__":
    main()
