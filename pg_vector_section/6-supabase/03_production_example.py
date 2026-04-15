"""
Lecture 6: Production-Ready RAG Service
A complete example with best practices for production
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()


# Configuration
@dataclass
class Config:
    # Database - use pooler URL in production
    database_url: str = os.getenv(
        "SUPABASE_DATABASE_URL",
        os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
        ),
    )
    collection_name: str = "production_documents"

    # Model settings
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.0

    # Search settings
    default_k: int = 5
    min_similarity: float = 0.5


class RAGService:
    """Production-ready RAG service with pgvector"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._vectorstore = None
        self._chain = None

    @property
    def vectorstore(self) -> PGVector:
        """Lazy initialization of vectorstore"""
        if self._vectorstore is None:
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            self._vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.config.collection_name,
                connection=self.config.database_url,
                use_jsonb=True,
            )
        return self._vectorstore

    @property
    def chain(self):
        """Lazy initialization of RAG chain"""
        if self._chain is None:
            self._chain = self._create_chain()
        return self._chain

    def _create_chain(self):
        """Create the RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.default_k}
        )

        llm = ChatOpenAI(
            model=self.config.chat_model, temperature=self.config.temperature
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer concisely and accurately. If the context doesn't contain relevant information,
say "I don't have enough information to answer that question."
"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vectorstore"""
        return self.vectorstore.add_documents(documents)

    def search(
        self, query: str, k: Optional[int] = None, filter_dict: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Search with optional filtering"""
        search_kwargs = {"k": k or self.config.default_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        return self.vectorstore.similarity_search_with_score(query, **search_kwargs)

    def ask(self, question: str) -> str:
        """Ask a question using RAG"""
        return self.chain.invoke(question)

    def ask_with_sources(self, question: str) -> dict:
        """Ask a question and return sources"""
        # Get relevant documents
        docs_with_scores = self.search(question)

        # Generate answer
        answer = self.ask(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity": score,
                }
                for doc, score in docs_with_scores
            ],
        }


def main():
    print("=" * 60)
    print("Production RAG Service Demo")
    print("=" * 60)

    # Initialize service
    print("\n🚀 Initializing RAG service...")
    service = RAGService()
    print("✅ Service ready!")

    # Add sample documents
    print("\n📄 Adding sample documents...")
    sample_docs = [
        Document(
            page_content="pgvector is a PostgreSQL extension for vector similarity search. It supports exact and approximate nearest neighbor search.",
            metadata={"source": "docs", "topic": "pgvector"},
        ),
        Document(
            page_content="HNSW indexes provide fast approximate nearest neighbor search with configurable accuracy through ef_search parameter.",
            metadata={"source": "docs", "topic": "indexing"},
        ),
        Document(
            page_content="For production deployments, use connection pooling to efficiently manage database connections and avoid exhausting limits.",
            metadata={"source": "best-practices", "topic": "production"},
        ),
    ]

    ids = service.add_documents(sample_docs)
    print(f"✅ Added {len(ids)} documents")

    # Test search
    print("\n🔍 Testing search...")
    results = service.search("How do I make pgvector faster?", k=2)
    for doc, score in results:
        print(f"   Score: {score:.4f} - {doc.page_content[:50]}...")

    # Test RAG
    print("\n❓ Testing RAG...")
    question = "What is pgvector and how do I use it?"
    response = service.ask_with_sources(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nSources used: {len(response['sources'])}")

    print("\n" + "=" * 60)
    print("✅ Production RAG service demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
