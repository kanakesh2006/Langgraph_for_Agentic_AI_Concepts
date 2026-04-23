from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
import operator

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(model="llama-3.1-8b-instant")


# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:

  """
  Peform a basic arithmetic operation on two numbers.
  Supported operations : add, sub, mul, div
  """

  try:

    if operation == "add":
      result = first_num + second_num
    elif operation == "sub":
      result = first_num - second_num
    elif operation == "mul":
      result = first_num * second_num
    elif operation == "div":
      if second_num == 0:
        return {"error": "Division by aero is not allowed"}
      result = first_num / second_num
    else:
      return {"error": f"Unsupported operation '{operation}'"}

    return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}

  except Exception as e:
    return {"error": str(e)}



@tool
def get_stock_price(symbol: str) -> dict:
  """
  Fetch latest stock price for a given symbol (e.g 'AAPL', 'TSLA')
  using Alpha Vantage with API key in the url
  """

  # https://www.alphavantage.co/documentation/
  # api key : WYNMZDP4MW6VNAGU
  url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=WYNMZDP4MW6VNAGU"
  r = requests.get(url)
  data = r.json()

  return data



# Make tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)


# state
class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


# graph nodes
def chat_node(state: ChatState):
  """
  LLM node that may answer or request a tool call.
  """

  messages = state['messages']
  response = llm_with_tools.invoke(messages)
  return {"messages": [response]}

tool_node = ToolNode(tools)


# sqlite3 database connection
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False )
checkpointer = SqliteSaver(conn = conn)


# graph structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node('tools', tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpointer)

# chatbot


# helper function
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)


