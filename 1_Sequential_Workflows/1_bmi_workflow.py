from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Define state
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category: str

def calculate_bmi(state: BMIState) -> BMIState:
    weight = state['weight_kg']
    height = state['height_m']
    bmi = weight / (height**2)
    state['bmi'] = round(bmi, 2)
    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif 18.5 <= bmi < 25:
        state['category'] = "Normal"
    elif 25 <= bmi < 30:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obese"
    return state

# Define you graph
graph = StateGraph(BMIState)

# Add nodes to your graph
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi", label_bmi)

# Add edges to your graph
graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', "label_bmi")
graph.add_edge("label_bmi", END)

# Compile your graph
workflow = graph.compile()

initial_state = {"weight_kg": 93, "height_m": 1.93}
# Execute your graph
final_state = workflow.invoke(initial_state)
print(final_state)

workflow.get_graph().print_ascii()

