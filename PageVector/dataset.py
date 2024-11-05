from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


embeddings = OpenAIEmbeddings()


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

def search_similar_documents(query, top_k=5):
    similar_docs = db.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, score in similar_docs:
        results.append({"document": doc, "relevance_score": score})
    return results


def generate_response(context, query):
    
    prompt = f"""
    You are a very enthusiastic and helpful assistant! Given the following sections, answer the question using only that information. 

    Context sections:
    {context}

    Question: {query}

    Answer in markdown format :
    """

    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0
    )

    
    return response.choices[0].message.content


print(contexto)

print(generate_response(similar_docs,query))