# Vector Database Masterclass

**Complete course on vector databases, embeddings, and RAG pipelines.**

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/pdichone/vector-databases-course.git
cd vector-databases-course
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Get your API keys
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

---

## Last Updated
April 2026 - All code tested with latest APIs
