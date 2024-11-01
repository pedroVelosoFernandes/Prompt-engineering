from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import openai
from openai import OpenAI
import os
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
query = "Make codes in python beeing a solution to the problem" #minimum
similar_docs = db.similarity_search_with_score(query, k=5)

contexto = ""
# Exibir documentos similares
for doc, score in similar_docs:
    contexto += f"Documento:\n{doc}\nRelevância: {score}\n\n"

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

def search_similar_documents(query, top_k=5):
    similar_docs = db.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, score in similar_docs:
        results.append({"document": doc, "relevance_score": score})
    return results

# Função para criar prompt e obter resposta do ChatGPT
def generate_response(context, query):
    # Cria o prompt completo com o contexto e a consulta
    prompt = f"""
    You are a very enthusiastic and helpful assistant! Given the following sections, answer the question using only that information. 

    Context sections:
    {context}

    Question: {query}

    Answer in markdown format :
    """

    # Usa openai.ChatCompletion.create para gerar a resposta
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Use "gpt-4" se você tiver acesso
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0
    )

    # Retorna a resposta gerada pelo modelo
    return response


print(contexto)

print(generate_response(similar_docs,query))