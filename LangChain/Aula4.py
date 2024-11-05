from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv, find_dotenv
import os
import Contexto
import json

# Carrega as variáveis de ambiente
_ = load_dotenv(find_dotenv())
set_debug(True)

# Variáveis de contexto
descricao_de_testes = Contexto.descricao_de_testes
user_case = Contexto.user_case
contexto = Contexto.contexto

llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.5,
    api_key=os.environ['OPENAI_API_KEY']
)

class Descricao(BaseModel):
    descricao: str = Field(description="Descrição dos testes a serem gerados")

class Codigo(BaseModel):
    
    codigo: str = Field(description="Código Python gerado com base na descrição")

class Testes(BaseModel):
    testes: str = Field(description="Bateria de testes gerados com base na descrição e código")

class Asserts(BaseModel):
    asserts: str = Field(description="Asserts gerados para os testes")

descricao_parser = JsonOutputParser(pydantic_object=Descricao)
codigo_parser = JsonOutputParser(pydantic_object=Codigo)
testes_parser = JsonOutputParser(pydantic_object=Testes)
asserts_parser = JsonOutputParser(pydantic_object=Asserts)

modelo_da_descricao = ChatPromptTemplate.from_template(
    "crie uma {descricao_de_testes} com base na {user_case} e no {contexto}.\n\n{formatacao_de_saida}",
    partial_variables={"formatacao_de_saida": descricao_parser.get_format_instructions()}
)

modelo_dos_codigos = ChatPromptTemplate.from_template(
    "crie o código Python necessário com base na {descricao}. O código deve ser executável e refletir todos os aspectos mencionados.Observacao: nao utilize acentos na linguagem natural\n\n{formatacao_de_saida}",
    partial_variables={"formatacao_de_saida": codigo_parser.get_format_instructions()}
)

modelo_dos_testes = ChatPromptTemplate.from_template(
    """crie testes unittest em python executáveis com base na {descricao_de_testes} e no {codigo}. \
    Os testes devem ser executáveis com "exec()" em Python e devem cobrir os diferentes caminhos de execução do código.\n\n{formatacao_de_saida}""",
    partial_variables={"formatacao_de_saida": testes_parser.get_format_instructions()}
)

modelo_dos_asserts = ChatPromptTemplate.from_template(
    """crie asserts para testar se os {testes} criados passam quando devem passar e \
    falham quando devem falhar. Exemplo: assert foo == bar.\n\n{formatacao_de_saida}""",
    partial_variables={"formatacao_de_saida": asserts_parser.get_format_instructions()}
)

descricao_prompt = modelo_da_descricao.format(
    descricao_de_testes=descricao_de_testes, user_case=user_case, contexto=contexto
)
descricao_saida = llm.invoke(descricao_prompt).content
descricao_structured = descricao_parser.parse(descricao_saida)

codigo_prompt = modelo_dos_codigos.format(descricao=descricao_structured['descricao'])
codigo_saida = llm.invoke(codigo_prompt).content
codigo_structured = codigo_parser.parse(codigo_saida)

testes_prompt = modelo_dos_testes.format(descricao_de_testes=descricao_structured['descricao'], codigo=codigo_structured['codigo'])
testes_saida = llm.invoke(testes_prompt).content
testes_saida = testes_saida.replace('"""', '"')
testes_structured = testes_parser.parse(testes_saida)

asserts_prompt = modelo_dos_asserts.format(testes=testes_structured['testes'])
asserts_saida = llm.invoke(asserts_prompt).content
asserts_structured = asserts_parser.parse(asserts_saida)

print("Descrição:")
print(descricao_structured)

print("\nCódigo:")
print(codigo_structured)

print("\nTestes:")
print(testes_structured)

print("\nAsserts:")
print(asserts_structured)

descricao_texto = descricao_structured['descricao']
with open("descricao_gerada.txt", "w") as file:
    file.write(descricao_texto)

codigo_python = codigo_structured['codigo']
with open("codigo_gerado.py", "w") as file:
    file.write(codigo_python)

testes_codigo = testes_structured['testes']
with open("test_gerados_test.py", "w") as file:
    file.write(testes_codigo)

asserts_texto = asserts_structured['asserts']
with open("asserts_gerados.txt", "w") as file:
    file.write(asserts_texto)

# Salvando os arquivos JSON
with open("descricao_gerada.json", "w") as file:
    json.dump(descricao_structured, file, indent=4)

with open("codigo_gerado.json", "w") as file:
    json.dump(codigo_structured, file, indent=4)

with open("testes_gerados.json", "w") as file:
    json.dump(testes_structured, file, indent=4)

with open("asserts_gerados.json", "w") as file:
    json.dump(asserts_structured, file, indent=4)