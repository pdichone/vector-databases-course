import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model

load_dotenv()


model = init_chat_model("gpt-4o-mini")


# Load documents with langchain
loader = DirectoryLoader(
    path="./data/new_articles", glob="**/*.txt", loader_cls=TextLoader
)
document = loader.load()

# print(document)

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(document)
print(len(texts))

# print(texts[0])

# Generate embeddings and store in vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = "./db/chroma_db_real_world"
vectordb = Chroma.from_documents(
    documents=texts, embedding=embeddings, persist_directory=persist_directory
)  # This will create the Chroma object and persist the embeddings to the directory


# Now we can query the Chroma object for similar sentences
retriever = vectordb.as_retriever()

# res = retriever.invoke("How much did microsoft raise?")
# print(res)


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

response = rag_chain.invoke("talk about databricks news.")
print(response)
