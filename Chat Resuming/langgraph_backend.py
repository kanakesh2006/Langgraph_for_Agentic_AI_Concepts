
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
# from pydantic.v1.fields import FieldInfo as FieldInfoV1
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

# using  GROQ's  "llama-3.1-8b-instant"  model
# os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')

llm = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# response = chatbot.invoke(
#     {'messages': [HumanMessage(content = 'hello')]},
#     config = CONFIG
# )

# print(chatbot.get_state(config=CONFIG).values['messages'])