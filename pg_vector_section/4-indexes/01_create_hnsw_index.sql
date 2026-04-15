-- Lecture 4: HNSW Index
-- Recommended default for most use cases

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing index if any
DROP INDEX IF EXISTS documents_embedding_hnsw_idx;

-- Create HNSW index with default parameters
-- vector_cosine_ops = cosine distance (best for OpenAI embeddings)
CREATE INDEX documents_embedding_hnsw_idx ON documents
USING hnsw (embedding vector_cosine_ops);

-- OR: Create with custom parameters for better accuracy
CREATE INDEX documents_embedding_hnsw_idx ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,              -- Connections per node (default 16, higher = more accurate)
    ef_construction = 64 -- Build quality (default 64, higher = better index)
);

-- Check index size
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'documents';

-- Set search parameter (for queries)
-- Higher = more accurate but slower
SET hnsw.ef_search = 100;

-- Verify index is being used (run EXPLAIN on your query)
EXPLAIN ANALYZE
SELECT content
FROM documents
ORDER BY embedding <=> (
  SELECT array_agg(random())::vector(1536) 
  FROM generate_series(1, 1536)
)
LIMIT 5;


-- HNSW Characteristics:
-- ✅ Very fast queries
-- ✅ High accuracy (~99%+)
-- ✅ Handles updates well
-- ❌ Slow to build (hours for millions)
-- ❌ Higher memory usage
