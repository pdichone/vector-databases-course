-- Lecture 4: IVFFlat Index
-- Use when memory is constrained or data is mostly static

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing index if any
DROP INDEX IF EXISTS documents_embedding_ivfflat_idx;

-- Create IVFFlat index
-- lists = number of clusters (rule: sqrt(rows) to rows/1000)
-- For 1M rows: lists = 1000

-- First, check how many rows you have
SELECT COUNT(*) FROM documents;

-- Create index with appropriate lists value
-- For small datasets (< 1000 rows):
CREATE INDEX documents_embedding_ivfflat_idx ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

-- For medium datasets (1000-100K rows):
-- WITH (lists = 100);

-- For large datasets (100K-10M rows):
-- WITH (lists = 1000);

-- Check index size
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'documents';

-- Set probes for queries (default is 1, increase for accuracy)
SET ivfflat.probes = 10;

-- Verify index is being used
EXPLAIN ANALYZE
SELECT content
FROM documents
ORDER BY embedding <=> (
  SELECT array_agg(random())::vector(1536) 
  FROM generate_series(1, 1536)
)
LIMIT 5;

-- IVFFlat Characteristics:
-- ✅ Fast to build
-- ✅ Lower memory usage
-- ✅ Good for static data
-- ❌ Slower queries than HNSW
-- ❌ Lower accuracy (~95%)
-- ❌ Needs rebuild after many updates

-- When to rebuild IVFFlat:
-- After inserting >10% new data, consider:
REINDEX INDEX documents_embedding_ivfflat_idx;
