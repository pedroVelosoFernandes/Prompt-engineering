from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.pgvector import PGVector
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

descricao_de_testes = "crie uma descricao de testes que apresente o fluxo principal e os alternativos "
user_case = """Eu, enquanto administrador do sistema, quero utilizar o sistema para criar, \
editar e remover um estabelecimento. \
Um estabelecimento deverá possuir um código de acesso ao sistema (com 6 dígitos). \
O código de acesso deve ser informado sempre que se faz alguma operação enquanto \
estabelecimento. Se o código de acesso não for informado ou estiver incorreto, a \
operação irá obrigatoriamente falhar. Não há limite para o número de operações \
com inserção de código incorreto. \
"""
contexto = """Recentemente, diversas empresas do ramo alimentício têm se desvinculado \
dos grandes aplicativos de delivery. As causas dessa tendência são diversas e vão desde \
a transformação no modo de operação de cada estabelecimento, até às taxas abusivas das \
grandes plataformas. \
Porém, em 2024, simplesmente não é viável voltar ao modo de trabalho “pré-Ifood”... Foi \
por isso que a pizzaria Pits A decidiu desenvolver seu próprio aplicativo de delivery. \
E adivinha só… vocês foram escolhidos para ajudar! \
 """

llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.5,
    api_key= os.environ['OPENAI_API_KEY']
)

modelo_do_prompt = PromptTemplate.from_template(
    "crie uma {descricao_de_testes} com base na {user_case} e no {contexto}"
)
# atribuir os modelos para os valores
prompt = modelo_do_prompt.format(descricao_de_testes= descricao_de_testes,user_case=user_case,contexto= contexto)

resposta = llm.invoke(prompt)
print(resposta.content)