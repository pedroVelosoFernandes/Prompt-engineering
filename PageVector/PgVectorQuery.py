import os
import openai
import psycopg2
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Configuração do modelo de embeddings
embeddings = OpenAIEmbeddings()

# Conectar ao PostgreSQL e inserir os embeddings
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
# Confirmar a inserção e fechar a conexão
conn.commit()
cur.close()
conn.close()

print("Programa executado com sucesso")