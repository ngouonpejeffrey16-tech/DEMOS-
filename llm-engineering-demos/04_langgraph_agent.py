"""Demo 4 — A minimal LangGraph agent: StateGraph, tool node, conditional edge.

Graph shape:
    START -> agent -> (tool needed?) -> tools -> agent -> ... -> END

The State (a TypedDict) flows through every node; each node reads it and
returns a partial update. The conditional edge decides whether to loop
through the tool node or finish.
"""

import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.getenv("BASE_URL", "https://api.groq.com/openai/v1"),
)
MODEL = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression, e.g. '4500 * 0.2'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: unsupported characters."
    try:
        return str(eval(expression))  # safe here: input restricted above
    except Exception as error:
        return f"Error: {error}"


# --- 1. State: the shared memory that flows through the graph -------------
class AgentState(TypedDict):
    messages: list


# --- 2. Nodes: plain Python functions -------------------------------------
def agent_node(state: AgentState) -> AgentState:
    """LLM reads the conversation and decides: answer, or call a tool."""
    response = client.chat.completions.create(
        model=MODEL, messages=state["messages"], tools=TOOLS
    )
    message = response.choices[0].message
    return {"messages": state["messages"] + [message.model_dump()]}


def tool_node(state: AgentState) -> AgentState:
    """Execute every tool call requested by the last LLM message."""
    last = state["messages"][-1]
    results = []
    for call in last["tool_calls"]:
        arguments = json.loads(call["function"]["arguments"])
        output = calculator(**arguments)
        print(f"  [tool] calculator({arguments['expression']}) -> {output}")
        results.append(
            {"role": "tool", "tool_call_id": call["id"], "content": output}
        )
    return {"messages": state["messages"] + results}


# --- 3. Conditional edge: route based on the state -------------------------
def should_continue(state: AgentState) -> str:
    return "tools" if state["messages"][-1].get("tool_calls") else END


# --- 4. Build and compile the graph ----------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()

if __name__ == "__main__":
    question = (
        "A truck drives 4500 km per month. Maintenance costs 0.12 euro/km. "
        "What is the yearly maintenance budget?"
    )
    print(f"Question: {question}\n")
    final_state = app.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    print("\nAnswer:\n" + final_state["messages"][-1]["content"])
