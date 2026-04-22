from langgraph.graph import StateGraph, START, END
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatCohere()

class BlogState(TypedDict):
    title: str
    outline: str
    content: str

def create_outline(state: BlogState) -> BlogState:
    title = state['title']
    prompt = f"Generate a detailed outline for a blog on the topic - {title}"
    outline = model.invoke(prompt).content
    state['outline'] = outline
    return state

def create_blog(state: BlogState) -> BlogState:
    outline = state['outline']
    title = state['title']
    prompt = f"Generate a blog from the given topic - {title} and outline - {outline}"
    content = model.invoke(prompt).content
    state['content'] = content
    return state

graph = StateGraph(BlogState)

graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

initial_state = {"title": "Rise on AI in India"}

final_state = workflow.invoke(initial_state)
print(final_state)