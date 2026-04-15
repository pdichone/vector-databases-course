# Install dependencies
# pip install langchain langchain-openai langchain-chroma rank_bm25

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Documents with both semantic content AND specific identifiers
documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports "
        "gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the "
        "ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server "
        "rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. "
        "Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel "
        "at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have "
        "alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

print(f"Loaded {len(documents)} documents")


vectorstore = Chroma.from_documents(
    documents, embeddings, collection_name="hybrid_test"
)

# Create vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Return top 3

print("Vector retriever ready")


# BM25 works on the raw text
bm25_retriever = BM25Retriever.from_documents(documents, k=3)  # Return top 3

print("BM25 retriever ready")


# Hybrid retrieval using Reciprocal Rank Fusion (RRF)
# This is what EnsembleRetriever did internally
def hybrid_retrieve(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""
    doc_scores = {}  # page_content -> (score, doc)

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))
            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score, doc)
            else:
                doc_scores[key] = (rrf_score, doc)

    sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in sorted_docs[:k]]


print("Hybrid retriever ready")


def test_query(query, name, retriever):
    """Test a query and show results"""
    results = retriever.invoke(query)
    print(f'\\n{name} - Query: "{query}"')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"  {i+1}. {preview}")
    return results


# Test queries designed to challenge vector search
test_queries = [
    "SKU-7742X specifications",  # Exact product code
    "E_CONN_REFUSED error",  # Error code
    "How do I authenticate?",  # Semantic question
    "WCAG compliance",  # Acronym
    "router configuration",  # General semantic
]


for query in test_queries:
    print("=" * 60)

    # Vector only
    vector_results = test_query(query, "VECTOR", vector_retriever)

    # BM25 only
    bm25_results = test_query(query, "BM25", bm25_retriever)

    # Hybrid
    hybrid_results = hybrid_retrieve(
        query,
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )
    print(f'\nHYBRID - Query: "{query}"')
    for i, doc in enumerate(hybrid_results[:3]):
        preview = doc.page_content[:80] + "..."
        print(f"  {i+1}. {preview}")
