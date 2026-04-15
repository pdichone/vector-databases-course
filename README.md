# Vector Database Masterclass

**Complete course on vector databases, embeddings, and RAG pipelines — from fundamentals to production.**

> **Last Updated: April 2026** - Full course revamp with new sections on chunking strategies, hybrid search, pgvector/Supabase, and updated APIs (ChromaDB 1.5+, LangChain 1.2+, Pinecone 7.x, OpenAI latest models).

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/pdichone/vector-databases-course.git
cd vector-databases-course
```

### 2. Set up environment

#### Option A: Using uv (Recommended - Faster)
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option B: Using pip (Traditional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Get your API keys
- **OpenAI:** https://platform.openai.com/api-keys
- **Pinecone:** https://app.pinecone.io/

---

## Course Structure

### Core Files (Root)

| File | Section | Description |
|------|---------|-------------|
| `chroma_emb.py` | ChromaDB | Basic embedding functions with default embeddings |
| `chroma_persist.py` | ChromaDB | Persisting data to disk |
| `chroma_openai_emb.py` | ChromaDB | OpenAI embeddings with Chroma |
| `app.py` | ChromaDB | Quick-start demo with upsert and query |
| `raw_emb.py` | Embeddings | Raw OpenAI embeddings exploration |
| `vector_db_llm.py` | RAG | Complete RAG pipeline (no LangChain) |
| `vector_db_llm_lang_chain.py` | LangChain | RAG with LangChain + Chroma |
| `lang_test.py` | LangChain | LangChain `init_chat_model` and streaming demo |
| `pinecone_intro.py` | Pinecone | Pinecone basics |
| `pinecone_db_llm_lang_chain.py` | Pinecone | RAG with LangChain + Pinecone |

### Chunking Strategies (`chunking/`)

| File | Description |
|------|-------------|
| `semantic_chunking.py` | Semantic chunking with LangChain experimental — splits by meaning, not character count |
| `prod_ready.py` | Production-ready chunking comparing `RecursiveCharacterTextSplitter` vs `SemanticChunker` |

### Hybrid Search (`hybrid_search/`)

| File | Description |
|------|-------------|
| `prod_hybrid_search.py` | BM25 + vector search with weighted Reciprocal Rank Fusion (RRF) |
| `final_production.py` | Production-grade hybrid retrieval combining keyword and semantic search |

### pgvector + PostgreSQL (`pg_vector_section/`)

A full section on using PostgreSQL as a vector database with the pgvector extension.

| Folder | Topic | Files |
|--------|-------|-------|
| `1-setup/` | Docker setup, connections, enabling pgvector | `01_docker_setup.sh`, `02_basic_connection.py`, `03_enable_extension.sql`, `04_quick_demo.py`, `basic_data_types.sql` |
| `2-creating-querying/` | Creating tables, inserting vectors, querying | `01_create_table.sql`, `02_insert_vectors.py`, `03_similarity_search.py`, `04_metadata_filtering.py` |
| `3-similiarity-search/` | Similarity search deep dive with metadata filtering | `01_create_table.sql`, `02_insert_vectors.py`, `03_similarity_search.py`, `04_metadata_filtering.py` |
| `4-indexes/` | HNSW vs IVFFlat index comparison | `01_create_hnsw_index.sql`, `02_create_ivfflat_index.sql`, `03_index_comparison.py` |
| `5-langchain/` | LangChain + pgvector integration and full RAG chain | `01_setup_vectorstore.py`, `02_add_documents.py`, `03_similarity_search.py`, `04_rag_chain.py` |
| `6-supabase/` | Supabase as a managed pgvector backend | `01_supabase_connection.py`, `02_connection_pooling.py`, `03_production_example.py` |

---

## What's New (April 2026 Update)

### New Sections
- **Chunking Strategies** — Semantic chunking vs recursive character splitting with side-by-side comparison
- **Hybrid Search** — BM25 + vector search with weighted Reciprocal Rank Fusion (RRF) for better retrieval
- **pgvector (PostgreSQL)** — Full section: setup, CRUD, similarity search, HNSW/IVFFlat indexes, LangChain integration
- **Supabase** — Production-ready vector search using Supabase as a managed pgvector backend
- **LangChain LCEL patterns** — All LangChain files updated to use the modern `init_chat_model` and LCEL (LangChain Expression Language) patterns

### Updates
- Updated all ChromaDB code to v1.5+ API
- Updated LangChain imports to use `langchain-chroma` and `langchain-postgres` packages
- Fixed all deprecated OpenAI model references (now using `gpt-4o-mini`)
- Added `requirements.txt` with pinned dependencies
- Added `pyproject.toml` and `uv.lock` for reproducible builds
- All code tested and working with Python 3.13

---

## Prerequisites

| Requirement | Used In |
|-------------|---------|
| Python 3.10+ | All sections |
| OpenAI API key | Embeddings & LLM calls |
| Pinecone API key | Pinecone section only |
| Docker | pgvector section (for local PostgreSQL) |
| Supabase account | Supabase section only |

---

## Sample Data

The `data/new_articles/` directory contains sample text files for the RAG examples.

---

## Join The AI Guild

**This code is part of our AI community where entrepreneurs and programmers build real-world AI solutions.**

[Join the AI Guild Community](https://bit.ly/ai-guild-join)

### What You Get:
- Collaborate with like-minded builders
- Access to Code & Template Vault
- Guided learning paths
- Weekly live calls & Q&A
- Real-world AI projects
