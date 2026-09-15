"""
LangGraph orchestration of the Autonomous Data Analyst agent.

Flow:
    entry -> router -> {data_analysis | visualization | ml_anomaly | document_retrieval | general}
          -> each branch runs an LLM-with-tools loop (tool calling, not manual math)
          -> synthesize -> END

The router picks a branch so each branch's LLM call only sees the
tools relevant to it, which makes tool selection far more reliable
than exposing all tools everywhere.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.llm_factory import get_llm
from app.tools.data_tools import make_data_tools
from app.tools.ml_tools import make_ml_tools
from app.tools.rag_tools import make_rag_tools, pop_pending_sources
from app.tools.viz_tools import make_viz_tools, pop_pending_charts
from app.utils.logger import get_logger
from app.workflow.router import classify_route
from app.workflow.state import AgentState

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an autonomous data analyst assistant.

Rules you must follow:
1. NEVER compute statistics, aggregations, or numeric results yourself. Always call the
   appropriate tool and report the tool's actual returned numbers.
2. If a tool returns an error, explain the error to the user plainly -- do not make up a result.
3. Keep answers concise, concrete, and grounded only in tool outputs (and retrieved documents,
   when relevant). Cite specific numbers from the tool results.
4. If asked something that requires a tool you don't have in this turn, say so rather than
   guessing.
"""


def _build_branch_graph(session_id: str, tools: list) -> Any:
    """Build a small tool-calling subgraph for one branch."""
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        response = llm_with_tools.invoke([{"role": "system", "content": SYSTEM_PROMPT}, *messages])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
    else:
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)

    return graph.compile()


def run_agent_turn(session_id: str, user_text: str, history: list[dict] | None = None) -> dict[str, Any]:
    """
    Execute one full turn: route the request, run the matching
    tool-calling subgraph, and assemble a structured response.
    """
    llm = get_llm()
    route = classify_route(llm, user_text)
    logger.info("Session %s routed to '%s'", session_id, route)

    if route == "data_analysis":
        tools = make_data_tools(session_id)
    elif route == "visualization":
        tools = make_viz_tools(session_id)
    elif route == "ml_anomaly":
        tools = make_ml_tools(session_id)
    elif route == "document_retrieval":
        tools = make_rag_tools(session_id)
    else:
        tools = []

    branch_graph = _build_branch_graph(session_id, tools)

    messages = list(history or [])
    messages.append({"role": "user", "content": user_text})

    result_state = branch_graph.invoke({
        "session_id": session_id,
        "messages": messages,
        "route": route,
        "tool_results": {},
        "charts": [],
        "sources": [],
        "final_answer": None,
    })

    final_message = result_state["messages"][-1]
    answer_text = getattr(final_message, "content", str(final_message))

    charts = pop_pending_charts(session_id)
    sources = pop_pending_sources(session_id)

    return {
        "answer": answer_text,
        "route_taken": route,
        "charts": charts,
        "sources": sources,
        "messages": result_state["messages"],
    }
