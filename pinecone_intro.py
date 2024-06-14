import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

pinecone_key = os.getenv("PINCONE_API_KEY")

pc = Pinecone(api_key=pinecone_key)

pc.create_index(
    name="quickstart",
    dimension=8,  # Replace with your model dimensions
    metric="euclidean",  # Replace with your model metric
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
