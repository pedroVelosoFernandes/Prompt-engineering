import torch
# esse codigo aqui é o que Allan me mostrou que eu adptei
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2" 


embeddings = HuggingFaceEmbeddings(model_name=model_name)

instruction = "Given a web search query, retrieve relevant passages that answer the query:"
queries = [
    f"{instruction} How much protein should a female eat",
    f"{instruction} summit define",
]

query_embeddings = embeddings.embed_documents(queries)

for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
    print(f"Query {i + 1}: {query}")
    print(f"Embedding (primeiros 5 valores): {embedding[:5]}...\n")
    print(len(embedding))

torch.save(query_embeddings, "query_embeddings.pt")
embedding = embeddings.embed_query("abobora")
print(embedding)

print("Embeddings gerados e salvos com sucesso!")
