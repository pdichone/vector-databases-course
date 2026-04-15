"""
Lecture 5: Complete RAG Chain with pgvector
Demonstrates a full retrieval-augmented generation pipeline
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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


def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore, k: int = 3):
    """Create a RAG chain using LCEL"""

    # Components
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant answering questions about databases and AI.

Use ONLY the following context to answer the question. If you don't know
the answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
"""
    )

    # Build chain using LCEL (LangChain Expression Language)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def main():
    print("Setting up RAG chain...")
    vectorstore = get_vectorstore()
    chain = create_rag_chain(vectorstore)
    print("✅ RAG chain ready!\n")

    # Test questions
    questions = [
        "How do transformers process input sequences efficiently?",
        "What chunking strategies affect retrieval quality in RAG pipelines?",
        "How does IVFFlat indexing trade off recall for speed?",
        "What is hybrid search and how does it combine keyword and vector similarity?",
    ]

    for question in questions:
        print("=" * 60)
        print(f"❓ Question: {question}")
        print("-" * 60)

        answer = chain.invoke(question)
        print(f"💡 Answer: {answer}")
        print()

    print("=" * 60)
    print("✅ RAG chain demo complete!")


if __name__ == "__main__":
    main()
