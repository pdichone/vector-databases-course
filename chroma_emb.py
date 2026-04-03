from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

default_ef = DefaultEmbeddingFunction()

name = "Paulo"

emb = default_ef([name])

print(emb)
