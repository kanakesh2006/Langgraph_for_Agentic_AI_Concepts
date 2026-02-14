from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
# FIX: Import from langchain.agents, NOT langgraph
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

os.environ['LANGCHAIN PROJECT'] = " ReAct Agent"

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather data for a given city"""
    url = f'https://api.weatherstack.com{city}'
    response = requests.get(url)
    return str(response.json())

llm = ChatGroq(model="llama-3.1-8b-instant")
prompt = hub.pull("hwchase17/react")

# The Tools list
tools = [search_tool, get_weather_data]

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Wrap with AgentExecutor (This is the LangChain legacy way)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

if __name__ == "__main__":
    response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
    print(response['output'])
