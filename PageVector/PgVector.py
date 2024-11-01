import os
import openai
import psycopg2
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PythonLoader, DirectoryLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Configuração do modelo de embeddings
embeddings = OpenAIEmbeddings()

ds = load_dataset("google-research-datasets/mbpp", "full")
texts = [example['text'] for example in ds["train"]]

# Dividir os textos em trechos menores para documentação
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=50,length_function=len, separators=[",","."]
)

texts = text_splitter.create_documents(texts)

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

# Processar cada trecho de código para gerar embedding e inserir no banco
cur.execute("""
    CREATE TABLE IF NOT EXISTS codigo_embedding (
        id SERIAL PRIMARY KEY,
        codigo TEXT,
        embedding VECTOR(1536)
    );
""")


embeddings_list = []
for doc in texts:
    embeddings_list.append(embeddings.embed_query(doc.page_content))

for i in range(len(embeddings_list)):
    embedding = embeddings_list[i]
    content = texts[i].page_content
    cur.execute("INSERT INTO codigo_embedding (codigo,embedding) VALUES (%s,%s)",(content,embedding))
# Confirmar a inserção e fechar a conexão
conn.commit()
cur.close()
conn.close()

print("Códigos e embeddings inseridos na tabela com sucesso.")