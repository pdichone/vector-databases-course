-- Lecture 3: Create Tables for Vector Storage

-- Enable pgvector if not already done
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing table if it exists
DROP TABLE IF EXISTS documents;

-- Create documents table with vector column
-- 1536 dimensions = OpenAI text-embedding-3-small
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for faster similarity search (HNSW - recommended)
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops);

-- Verify table structure
\d documents

-- Alternative: Different embedding dimensions
-- OpenAI text-embedding-3-large: vector(3072)
-- Gemini embedding: vector(768)
-- BGE-small: vector(384)
