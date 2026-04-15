#!/bin/bash
# pgvector Docker Setup
# Run this to start PostgreSQL with pgvector

# Stop and remove existing container (if any)
docker stop pgvector 2>/dev/null
docker rm pgvector 2>/dev/null

# Start fresh container
docker run -d \
  --name pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16

echo "Waiting for PostgreSQL to start..."
sleep 3

# Verify it's running
docker exec pgvector psql -U postgres -c "SELECT version();"

echo ""
echo "✅ pgvector is running!"
echo ""
echo "Connection string: postgresql://postgres:postgres@localhost:5432/postgres"
echo ""
echo "To connect: psql -h localhost -U postgres"
