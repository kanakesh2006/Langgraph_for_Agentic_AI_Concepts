# from langchain_openai import ChatOpenAI
from pydantic.v1.fields import FieldInfo as FieldInfoV1
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

os.environ["LANGCHAIN_PROJECT"] = "Sequential LLM App"

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# model = ChatOpenAI()
model1 = ChatGroq(model="llama-3.1-8b-instant")

model2 = ChatGroq(model="llama-3.1-8b-instant")

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    "run_name": "sequential chain",
    "tags": ["llm app", "report generation", "summarization"],
    "metadata": {'model1': 'llama-3.1-8b-instant', 'model2': 'llama-3.1-8b-instant', 'model1_temp': 0.7, 'parser': 'StrOutputParser'}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config = config)

print(result)
