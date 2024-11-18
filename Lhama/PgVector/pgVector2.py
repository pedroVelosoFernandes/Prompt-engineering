from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from dotenv import load_dotenv, find_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2" 

embeddings = HuggingFaceEmbeddings(model_name=model_name)

ds = load_dataset("google-research-datasets/mbpp", "full")
texts = [example['text'] for example in ds["train"]]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=50,length_function=len, separators=[",","."]
)

texts = text_splitter.create_documents(texts)


CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/pgvector"
COLLECTION_NAME = 'text_documents_vectors'


db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)


query = "taks related with minimim number"
similar_docs = db.similarity_search_with_score(query, k=5)

contexto = ""

for doc, score in similar_docs:
    contexto += f"Documento:\n{doc}\nRelevância: {score}\n\n"

print(contexto)
