import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pinecone import Pinecone, ServerlessSpec


load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(api_key=openai_key, model="gpt-4o-mini")

pinecone_key = os.getenv("PINECONE_API_KEY")


# load documents
loader = DirectoryLoader(
    path="./data/new_articles/", glob="*.txt", loader_cls=TextLoader
)
document = loader.load()

# split text into sentences
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size=1000,
    chunk_overlap=20,
)
documents = text_splitter.split_documents(document)
print(f"Number of documents: {len(documents)}")

# get embeddings
embedding = OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")


pc = Pinecone(api_key=pinecone_key)

index_name = "tester-index"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )


index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=documents, embedding=embedding, index_name=index_name
)

# query = "tell me about writers strike"
# docs = docsearch.similarity_search(query)
# print(docs[0].page_content)


# Create a retriever
retriever = docsearch.as_retriever()


# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Build RAG chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response = rag_chain.invoke("tell me about databricks news.")

print("==== Answer ====")
print(response)
