-- Lecture 2: Enable pgvector Extension
-- Run this in psql or any PostgreSQL client

-- Enable the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it's installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Test the vector type
SELECT '[1.0, 2.0, 3.0]'::vector AS test_vector;

-- Check available operators
SELECT
    opfname AS operator_family,
    opcname AS operator_class
FROM pg_opclass
WHERE opcname LIKE 'vector%';
