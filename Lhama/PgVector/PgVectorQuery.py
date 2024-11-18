
import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2" 

embeddings = HuggingFaceEmbeddings(model_name=model_name)

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="pgvector",
    user="postgres",
    password="admin"
)
print("Conexão com sucesso!")
cur = conn.cursor()


query_embedding = embeddings.embed_query("Functions to find minimum cost path.")

cur.execute("""
    SELECT * FROM codigo_embedding ORDER BY embedding <-> %s::vector
    LIMIT 2;
""", (query_embedding,))

results = cur.fetchall()

for result in results:
    print(result[1])
conn.commit()
cur.close()
conn.close()

print("Programa executado com sucesso")