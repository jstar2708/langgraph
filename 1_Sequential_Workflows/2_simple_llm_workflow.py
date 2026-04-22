from langgraph.graph import StateGraph, START, END
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatCohere()


# Create a state
class LLMState(TypedDict):
    question: str
    answer: str


def llm_qa(state: LLMState) -> LLMState:
    # Extract the question
    question = state["question"]

    prompt = f"Answer the following question {question}"

    answer = model.invoke(prompt).content

    state["answer"] = answer
    return state


# Create graph
graph = StateGraph(LLMState)

# Add node
graph.add_node("llm_qa", llm_qa)

# Add edges
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

workflow = graph.compile()

inital_state = {"question": "How far is moon from the earth?"}

final_state = workflow.invoke(inital_state)
print(final_state)
