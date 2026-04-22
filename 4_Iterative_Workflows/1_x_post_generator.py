from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_cohere import ChatCohere
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import operator

load_dotenv()


# State
class TweetState(TypedDict):
    tweet: str
    topic: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iterations: int

    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]


class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final Evaluation result"
    )
    feedback: str = Field(description="Feedback for the tweet")


generator_llm = ChatCohere()
evaluator_llm = ChatCohere()
structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)
optimizer_llm = ChatCohere()


def generate_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(
            content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
"""
        ),
    ]

    # Send to generator llm

    response = generator_llm.invoke(messages).content

    return {"tweet": response, "tweet_history": [response]}


def evaluate_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(
            content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."
        ),
        HumanMessage(
            content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality - Is this fresh, or have you seen it a hundred times before?  
2. Humor - Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness - Is it short, sharp, and scroll-stopping?  
4. Virality Potential - Would people retweet or share it?  
5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "needs_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
"""
        ),
    ]

    response = structured_evaluator_llm.invoke(messages)
    return {"evaluation": response.evaluation, "feedback": response.feedback, "feedback_history": [response.feedback]}


def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(
            content="You punch up tweets for virality and humor based on given feedback."
        ),
        HumanMessage(
            content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
"""
        ),
    ]

    response = optimizer_llm.invoke(messages).content
    iteration = state["iteration"]
    return {"tweet": response, "iteration": iteration + 1, "tweet_history": [response]}


def route_evaluation(state: TweetState):
    if (
        state["evaluation"] == "approved"
        or state["iteration"] >= state["max_iterations"]
    ):
        return "approved"
    else:
        return "needs_improvement"


graph = StateGraph(TweetState)

graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("optimize_tweet", optimize_tweet)

graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")
graph.add_conditional_edges(
    "evaluate_tweet",
    route_evaluation,
    {"approved": END, "needs_improvement": "optimize_tweet"},
)
graph.add_edge("optimize_tweet", "evaluate_tweet")

workflow = graph.compile()

workflow.get_graph().print_ascii()

initial_state = {"topic": "Indian Railways", "iteration": 1, "max_iterations": 5}

final_state = workflow.invoke(initial_state)
print(final_state)
