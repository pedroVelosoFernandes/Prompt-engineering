#lcel
from operator import itemgetter
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

# Load environment variables
_ = load_dotenv(find_dotenv())
set_debug(True)

# Context variables
description = Contexto.test_description
user_case = Contexto.user_case
context = Contexto.context

input_dict = {"description": description, "user_case": user_case,"context": context}
# Initialize Language Model
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.0,
    api_key=os.environ['OPENAI_API_KEY']
)

# Define models with refined descriptions
class Description(BaseModel):
    description = Field(description="Generated test description detailing main and alternative flows for the specified use case")

class Code(BaseModel):
    code = Field(description="Executable Python code generated based on the test description")

class Tests(BaseModel):
    tests = Field(description="Python unittest cases covering primary and alternative execution paths based on the description and code")

class Asserts(BaseModel):
    asserts = Field(description="Python assert statements to validate unittest results, ensuring cases pass or fail as expected")

# Define parsers
description_parser = JsonOutputParser(pydantic_object=Description)
code_parser = JsonOutputParser(pydantic_object=Code)
tests_parser = JsonOutputParser(pydantic_object=Tests)
asserts_parser = JsonOutputParser(pydantic_object=Asserts)

# Define prompt templates with partial variables for output formatting
description_template = ChatPromptTemplate.from_template(
    "Generate a {description} based on the {user_case} and {context}. Include main and alternative flows.\n\n{output_format}",
    partial_variables={"output_format": description_parser.get_format_instructions()}
    
)

code_template = ChatPromptTemplate.from_template(
    "Gere o código Python necessário baseado na descrição do teste. Retorne **somente o JSON** no seguinte formato:\n\n{output_format}",
    partial_variables={"output_format": code_parser.get_format_instructions()}
)

tests_template = ChatPromptTemplate.from_template(
    """Create executable Python unittest cases based on {description} and the {code}. \
    Tests should be executable with 'exec()' in Python and cover diverse execution paths.\n\n{output_format}""",
    partial_variables={"output_format": tests_parser.get_format_instructions()}
)

asserts_template = ChatPromptTemplate.from_template(
    """Generate assert statements for the {tests}, verifying if they pass when expected and fail when necessary. Example: assert foo == bar.\n\n{output_format}""",
    partial_variables={"output_format": asserts_parser.get_format_instructions()}
)

# Corrige a construção da cadeia para utilizar templates corretamente
first = description_template | llm | description_parser
second = code_template | llm | code_parser
third = tests_template | llm | tests_parser
fourth = asserts_template | llm | asserts_parser

chain = (first | {"description": itemgetter("description"),"code":second} | third | fourth)


