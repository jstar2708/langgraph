from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

llm = ChatOllama(model='gemma3:4b', temperature=0)

def chat_node(state: ChatState):
    # Take user query
    messages = state['messages']

    # Pass query to LLM
    response = llm.invoke(messages)

    # Store response in state
    return {"messages": [response]}

# Add checkpointer
checkpointer = MemorySaver()

graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'

while True:
    user_query = input("Type here: ")
    if user_query.strip().lower() in ['exit', 'quit', 'bye']:
        break
    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke({"messages": [HumanMessage(content=user_query)]}, config=config)
    print("AI: ", response['messages'][-1].content)
    
