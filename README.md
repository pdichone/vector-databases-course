# Vector Database Masterclass

**Complete course on vector databases, embeddings, and RAG pipelines.**

> **Last Updated: April 2026** - All code updated and tested with the latest APIs including ChromaDB 1.5+, LangChain 1.2+, Pinecone 7.x, and OpenAI's newest models.

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

## Course Files

| File | Section | Description |
|------|---------|-------------|
| `chroma_emb.py` | ChromaDB | Basic embedding functions |
| `chroma_persist.py` | ChromaDB | Persisting data to disk |
| `chroma_openai_emb.py` | ChromaDB | OpenAI embeddings with Chroma |
| `vector_db_llm.py` | RAG | Complete RAG pipeline (no LangChain) |
| `vector_db_llm_lang_chain.py` | LangChain | RAG with LangChain + Chroma |
| `pinecone_intro.py` | Pinecone | Pinecone basics |
| `pinecone_db_llm_lang_chain.py` | Pinecone | RAG with LangChain + Pinecone |
| `raw_emb.py` | Embeddings | Raw OpenAI embeddings |

---

## What's New (April 2026 Update)

- Updated all ChromaDB code to v1.5+ API
- Updated LangChain imports to use `langchain-chroma` package
- Fixed all deprecated OpenAI model references (`gpt-4o-mini`)
- Added `requirements.txt` with pinned dependencies
- Added `pyproject.toml` for uv/modern Python tooling
- All code tested and working with Python 3.13

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
