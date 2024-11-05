from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chains import SequentialChain
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import Contexto

# Carrega as variáveis de ambiente
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Variáveis de contexto
descricao_de_testes = Contexto.descricao_de_testes
user_case = Contexto.user_case
contexto = Contexto.contexto

# Configuração do LLM
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.5,
    api_key=os.environ['OPENAI_API_KEY']
)

# Modelos de prompts
modelo_da_descricao = ChatPromptTemplate.from_template(
    "crie uma {descricao_de_testes} com base na {user_case} e no {context}"
)
modelo_dos_testes = ChatPromptTemplate.from_template(
    """crie uma bateria de testes com base na {descricao_de_testes}. \
    Devo conseguir executar a funcao "exec()" de python nesses testes. \
    Eles devem seguir a logica de fluxo, entao para cada caminho de execução, devo \
    ser capaz de avaliar.
    """
)
modelo_dos_asserts = ChatPromptTemplate.from_template(
    """crie asserts para testar se os {testes} criados passam quando devem passar e \
        falham quando devem falhar. Basicamente seriam os parametros passados para a \
        funcao.
        exemplo: assertNotTrue(true,primo(5))"""
)

# Criando a sequência de prompts e LLMs usando o pipe (`|`) com RunnableSequence
cadeia = (
    modelo_da_descricao | llm |
    modelo_dos_testes | llm |
    modelo_dos_asserts | llm
)

# Invocando a cadeia com o input correto
resultado = cadeia.invoke({
    "context": contexto,
    "descricao_de_testes": descricao_de_testes,
    "user_case": user_case
})

print(resultado)
