import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
import os
from PyPDF2 import PdfReader


model_name = "sentence-transformers/all-MiniLM-L6-v2" 

embeddings = HuggingFaceEmbeddings(model_name=model_name)

def process_pdfs_to_embeddings(pdf_folder):
    """
    Processa PDFs em uma pasta, gerando embeddings para os títulos e conteúdos.

    Args:
        pdf_folder (str): Caminho para a pasta contendo os PDFs.

    Returns:
        tuple: (content_embeddings, title_embeddings, titles)
    """
    content_embeddings = []
    title_embeddings = []
    titles = []

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)

            pdf_reader = PdfReader(file_path)
            title = file_name.replace(".pdf", "") 
            content = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

            
            title_embedding = embeddings.embed_query(title)
            content_embedding = embeddings.embed_query(content)

            
            titles.append(title)
            title_embeddings.append(title_embedding)
            content_embeddings.append(content_embedding)

    return content_embeddings, title_embeddings, titles

conn = psycopg2.connect(
    host="localhost",
    port="5433",
    dbname="pgvector",
    user="postgres",
    password="admin",
    options='-c client_encoding=UTF8'
)
print("Conexão com sucesso!")
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS tabela_embedding (
        id SERIAL PRIMARY KEY,
        titulo TEXT,
        tituloEmb VECTOR(384),
        conteudoEmb VECTOR(384)
    );
""")

content_embeddings, title_embeddings, titles = process_pdfs_to_embeddings("docs-sample")
for i in range(len(titles)):
    title_embedding = title_embeddings[i]
    content_embedding = content_embeddings[i]
    title = titles[i]
    cur.execute("INSERT INTO tabela_embedding (titulo,tituloEmb,conteudoEmb) VALUES (%s,%s,%s)",(title,title_embedding,content_embedding))

conn.commit()
cur.close()
conn.close()
print("Códigos e embeddings inseridos na tabela com sucesso.")
