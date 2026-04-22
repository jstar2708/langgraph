from langgraph.graph import StateGraph, START, END
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

load_dotenv()

model = ChatCohere()


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay.")
    score: int = Field(description="Score out of 10", ge=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)


class UPSEState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[str], operator.add]
    avg_score: float


def evaluate_language(state: UPSEState) -> UPSEState:
    essay = state["essay"]
    prompt = f"Evaluate the language quality of the following essay and provide a feedback assign a score out of 10.\n{essay}"
    result = structured_model.invoke(prompt)
    return {"language_feedback": result.feedback, "individual_scores": [result.score]}


def evaluate_analysis(state: UPSEState) -> UPSEState:
    essay = state["essay"]
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback assign a score out of 10.\n{essay}"
    result = structured_model.invoke(prompt)
    return {"analysis_feedback": result.feedback, "individual_scores": [result.score]}


def evaluate_clarity(state: UPSEState) -> UPSEState:
    essay = state["essay"]
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback assign a score out of 10.\n{essay}"
    result = structured_model.invoke(prompt)
    return {"clarity_feedback": result.feedback, "individual_scores": [result.score]}


def final_evaluation(state: UPSEState) -> UPSEState:
    essay = state["essay"]
    prompt = f"Based on the following feedbacks create a summarized feedback.\n Language Feedback: {state['language_feedback']}\nClarity Feedback: {state['clarity_feedback']}\nAnalysis Feedback: {state['analysis_feedback']}\n "
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])
    return {"overall_feedback": overall_feedback, "avg_score": avg_score}


graph = StateGraph(UPSEState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)


graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

workflow.get_graph().print_ascii()

initial_state = {
    "essay": """
A mother is often the silent architect of a person’s foundation, providing a unique blend of unwavering support and selfless love. Beyond the biological bond, motherhood represents a sanctuary of emotional resilience and guidance. She is the first teacher, instilling values and confidence while navigating the complexities of caregiving. Her influence transcends words, manifesting in the small sacrifices and the profound encouragement that shape one's character. In a rapidly changing world, a mother remains a constant—a source of strength that empowers her children to face challenges with grace. She is, quite simply, the heartbeat of the home.
"""
}

final_state = workflow.invoke(initial_state)
print(final_state)