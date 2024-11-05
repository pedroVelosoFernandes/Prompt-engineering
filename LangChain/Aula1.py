from dotenv import load_dotenv, find_dotenv
import os
import openai
from openai import OpenAI

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
prompt = f"crie uma {descricao_de_testes} com base na {user_case} e no {contexto}"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
resposta = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0
    )

print(resposta.choices[0].message.content)