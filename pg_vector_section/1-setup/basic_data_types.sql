-- A 3-dimensional vector
SELECT '[1.0, 2.0, 3.0]'::vector;

-- Specify dimensions in table definition
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)  -- OpenAI dimensions
);