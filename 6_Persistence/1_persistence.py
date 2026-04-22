from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(model="gemma3:4b")


class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


graph = StateGraph(JokeState)


def generate_joke(state: JokeState):
    prompt = f"Generate a joke on the topic {state['topic']}"
    result = llm.invoke(prompt).content
    return {"joke": result}


def explain_joke(state: JokeState):
    prompt = f"Explain the following joke {state['joke']}"
    result = llm.invoke(prompt).content
    return {"explanation": result}


graph.add_node("generate_joke", generate_joke)
graph.add_node("explain_joke", explain_joke)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "explain_joke")
graph.add_edge("explain_joke", END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config_1 = {"configurable": {"thread_id": 1}}

result = workflow.invoke({"topic": "Cricket"}, config=config_1)

print(workflow.get_state(config_1), end="\n\n")

print(list(workflow.get_state_history(config_1)), end="\n\n")

config_2 = {"configurable": {"thread_id": 2}}

result = workflow.invoke({"topic": "Pizza"}, config=config_2)

print(workflow.get_state(config_2), end="\n\n")

print(list(workflow.get_state_history(config_2)), end="\n\n")
