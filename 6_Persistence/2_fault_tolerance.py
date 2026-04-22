from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver


class CrashState(TypedDict):
    input: str
    step_1: str
    step_2: str
    step_3: str


def execute_step_1(state: CrashState):
    print("Step 1 executed successfully")
    return {"step_1": "done"}


def execute_step_2(state: CrashState):
    print("Simulating crash!....")
    from time import sleep

    sleep(10)
    print("Step 2 executed successfully")
    return {"step_2": "done"}


def execute_step_3(state: CrashState):
    print("Step 3 executed successfully")
    return {"step_3": "done"}


try:

    graph = StateGraph(CrashState)

    graph.add_node("step_1", execute_step_1)
    graph.add_node("step_2", execute_step_2)
    graph.add_node("step_3", execute_step_3)

    graph.add_edge(START, "step_1")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_3")
    graph.add_edge("step_3", END)

    checkpointer = InMemorySaver()

    workflow = graph.compile(checkpointer=checkpointer)
    workflow.invoke({"input": "start"}, config={"configurable": {"thread_id": 1}})
except KeyboardInterrupt:
    print("App crashed!!!")

print(workflow.get_state({"configurable": {"thread_id": 1}}))

workflow.invoke(
    None,       # Pass None so that it resumes execution and not restarts it.
    {"configurable": {"thread_id": 1}},
)
