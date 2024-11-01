from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# Configurações iniciais
import os
import openai
from dotenv import load_dotenv, find_dotenv

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Configuração do modelo de embeddings
embeddings = OpenAIEmbeddings()

# Carregar o dataset e extrair os textos
ds = load_dataset("google-research-datasets/mbpp", "full")
texts = [example['text'] for example in ds["train"]]

# Dividir os textos em trechos menores para documentação
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=50,length_function=len, separators=[",","."]
)

texts = text_splitter.create_documents(texts)

# Conexão com o banco de dados
CONNECTION_STRING = "postgresql+psycopg2://postgres:admin@localhost:5432/pgvector"
COLLECTION_NAME = 'text_documents_vectors'

# Carregar documentos e embeddings para o banco de dados
db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Consulta por similaridade
query = "Functions to find lolwest cost path" #minimum
similar_docs = db.similarity_search_with_score(query, k=5)

# Exibir documentos similares
for doc, score in similar_docs:
    print(f"Documento:\n{doc}\nRelevância: {score}\n\n")

# benefits with LLMs

""" So we have a data-store containing all our vectors. This can be used when we have a lot of context, 
and want to only select the most relevant or similar chunks as context when querying and prompting language-models.

Language models tend to have a context length or context window, that limits the number of 
tokens they can consider at a time. 

If we upload a repository of 1000 documents, we cannot realistically pass all that context in 
a call to the language-model. The vector database allows us to select chunks that closely match 
what we've provided in the prompt, and only pass certain segments of the text.

This is powerful, as it allows us to take advantage of greater amounts of text content. We can 
embed a large corpus of documents into chunks, and can select the relevant segments in our code. """