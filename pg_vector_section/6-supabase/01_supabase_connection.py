"""
Lecture 6: Supabase Connection
Demonstrates connecting to Supabase's managed pgvector
"""

import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Supabase connection string format:
# postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
#
# Get these from: Supabase Dashboard → Project Settings → Database

SUPABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

# For demo, use local if Supabase not configured
DATABASE_URL = SUPABASE_URL or os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)


def connect_to_supabase():
    """Connect to Supabase pgvector"""

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Note: Supabase has pgvector pre-installed
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="production_docs",
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def verify_connection(vectorstore):
    """Verify the connection works"""
    from langchain_core.documents import Document

    # Add a test document
    test_doc = Document(
        page_content="This is a test document to verify Supabase connection.",
        metadata={"test": True},
    )

    try:
        ids = vectorstore.add_documents([test_doc])
        print(f"✅ Added test document: {ids[0]}")

        # Search for it
        results = vectorstore.similarity_search("test document", k=1)
        if results:
            print(f"✅ Search works: {results[0].page_content[:50]}...")

        # Clean up
        vectorstore.delete(ids)
        print("✅ Cleanup complete")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Supabase pgvector Connection Test")
    print("=" * 60)

    if SUPABASE_URL:
        print(f"\n🔗 Connecting to Supabase...")
        print(
            f"   Host: {SUPABASE_URL.split('@')[1].split(':')[0] if '@' in SUPABASE_URL else 'configured'}"
        )
    else:
        print("\n⚠️  SUPABASE_DATABASE_URL not set")
        print("   Using local PostgreSQL instead")
        print("\n   To use Supabase:")
        print("   1. Create project at supabase.com")
        print("   2. Go to Settings → Database")
        print("   3. Copy connection string")
        print("   4. Set SUPABASE_DATABASE_URL in .env")

    vectorstore = connect_to_supabase()
    print("\n✅ Connected!")

    print("\n🧪 Running verification...")
    success = verify_connection(vectorstore)

    if success:
        print("\n✅ Supabase pgvector is ready for production!")
    else:
        print("\n❌ Verification failed - check your connection")


if __name__ == "__main__":
    main()
