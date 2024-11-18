import psycopg2
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2" 

embeddings = HuggingFaceEmbeddings(model_name=model_name)

ds = load_dataset("google-research-datasets/mbpp", "full")
texts = [example['text'] for example in ds["test"]]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=50,length_function=len, separators=[",","."]
)

texts = text_splitter.create_documents(texts)

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    dbname="pgvector",
    user="postgres",
    password="admin",
    options='-c client_encoding=UTF8'
)
print("Conexão com sucesso!")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS codigo_embedding (
        id SERIAL PRIMARY KEY,
        codigo TEXT,
        embedding VECTOR(384)
    );
""")


embeddings_list = []
for doc in texts:
    embeddings_list.append(embeddings.embed_query(doc.page_content))

for i in range(len(embeddings_list)):
    embedding = embeddings_list[i]
    content = texts[i].page_content
    cur.execute("INSERT INTO codigo_embedding (codigo,embedding) VALUES (%s,%s)",(content,embedding))

conn.commit()
cur.close()
conn.close()
print("Códigos e embeddings inseridos na tabela com sucesso.")