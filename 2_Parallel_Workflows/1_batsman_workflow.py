from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percent: float
    summary: str

def calculate_sr(state: BatsmanState) -> BatsmanState:
    sr = (state['runs']/state['balls']) * 100
    return {"sr": sr}

def calculate_bpb(state: BatsmanState) -> BatsmanState:
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {"bpb": bpb}

def calculate_boundary_percent(state: BatsmanState) -> BatsmanState:
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']) * 100
    return {"boundary_percent": boundary_percent}

def summary(state: BatsmanState) -> BatsmanState:
    summary = f"""
        Strike rate : {state['sr']}
        Boundary per ball : {state['bpb']}
        Boundary percentage : {state['boundary_percent']}
"""
    return {"summary": summary}

graph = StateGraph(BatsmanState)

graph.add_node("calculate_sr", calculate_sr)
graph.add_node("calculate_bpb", calculate_bpb)
graph.add_node("calculate_boundary_percent", calculate_boundary_percent)
graph.add_node("summary", summary)

graph.add_edge(START, 'calculate_sr')
graph.add_edge(START, 'calculate_bpb')
graph.add_edge(START, 'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()
initial_state = {
    "runs": 18000,
    "balls": 15647,
    "fours": 1340,
    "sixes": 889
}

workflow.get_graph().print_ascii()

final_state = workflow.invoke(initial_state)
print(final_state)