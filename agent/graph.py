"""
LangGraph Agentic RAG Graph for Enterprise Multi-Modal Financial Intelligence.

Features:
- StateGraph with checkpointer (MemorySaver) for state persistence and time-travel in LangGraph Studio.
- RAG Retrieval Tool: Qdrant Hybrid Search (Dense Qwen 1024d + Sparse FastEmbed BM25 + RRF Fusion), 10 chunks per query.
- Multi-rewrite query execution allowing the agent to retrieve multiple angles (tables, narrative, figures).
- Grounded Synthesis Tool: High-capacity LLM (Groq Llama 3.3 70B / equivalent) with exact page citations and footnote reconciliation.
- Agent Orchestration LLM: Groq Llama 3.1 8B Instant (with automatic fallback to gpt-oss-20b if unavailable).
"""

from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agent.models import get_agent_llm
from agent.tools import retrieve_chunks, generate_grounded_answer
from agent.prompt import AGENT_SYSTEM_PROMPT

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieval_count: int


# 2. Initialize Models & Checkpointer
checkpointer = MemorySaver()
agent_llm = get_agent_llm()
available_tools = {t.name: t for t in [retrieve_chunks, generate_grounded_answer]}


# 3. Graph Node Definitions
def agent_node(state: AgentState):
    """Agent decision node: analyzes inquiry, plans query rewrites, and invokes tools."""
    msgs = state["messages"]
    count = state.get("retrieval_count", 0)

    # Dynamic prompt guidance to prevent runaway loops
    if count >= 2:
        prompt_text = (
            AGENT_SYSTEM_PROMPT
            + "\n\nCRITICAL: You have gathered multi-angle retrieval evidence (2 passes completed). "
            "You MUST now invoke 'generate_grounded_answer' to synthesize the final citation-backed response!"
        )
    else:
        prompt_text = AGENT_SYSTEM_PROMPT

    model = agent_llm.bind_tools([retrieve_chunks, generate_grounded_answer])
    system = SystemMessage(content=prompt_text)
    response = model.invoke([system] + msgs)
    return {"messages": [response]}


def tool_node(state: AgentState):
    """Tool execution node: executes Qdrant hybrid retrieval or grounded synthesis."""
    last_msg = state["messages"][-1]
    count = state.get("retrieval_count", 0)
    tool_messages = []

    for tc in getattr(last_msg, "tool_calls", []):
        t_name = tc["name"]
        t_args = tc["args"]
        t_id = tc["id"]

        if t_name in available_tools:
            tool = available_tools[t_name]
            result = tool.invoke(t_args)
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=t_id, name=t_name))
            if t_name == "retrieve_chunks":
                count += 1

    return {"messages": tool_messages, "retrieval_count": count}


def synthesize_node(state: AgentState):
    """Guaranteed synthesis node: aggregates evidence and produces grounded executive answer."""
    # Find original user query
    user_q = ""
    for m in state["messages"]:
        if getattr(m, "type", "") == "human":
            user_q = m.content
            break
    if not user_q:
        user_q = state["messages"][0].content

    # Collect all retrieved evidence from tools
    evidence_pieces = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.name == "retrieve_chunks":
            evidence_pieces.append(m.content)

    combined_evidence = "\n\n".join(evidence_pieces)
    grounded_ans = generate_grounded_answer.invoke({
        "question": user_q,
        "retrieved_evidence": combined_evidence
    })

    return {"messages": [AIMessage(content=str(grounded_ans))]}


# 4. Conditional Edge Routing
def should_continue(state: AgentState):
    """Determine whether to invoke tools or conclude."""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


def after_tools(state: AgentState):
    """Route after tool execution: check if synthesis finished or if we reached retrieval budget."""
    last_msg = state["messages"][-1]
    # If generate_grounded_answer was explicitly invoked by agent, we're done
    if getattr(last_msg, "name", "") == "generate_grounded_answer":
        return END

    # If the agent has retrieved 2+ times, trigger the synthesis node
    if state.get("retrieval_count", 0) >= 2:
        return "synthesize"

    # Otherwise allow agent to perform another rewrite
    return "agent"


# 5. Build and Compile Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("synthesize", synthesize_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_conditional_edges("tools", after_tools, ["agent", "synthesize", END])
workflow.add_edge("synthesize", END)

# Export compiled graph for LangGraph Studio (persistence is handled automatically by LangGraph API)
graph = workflow.compile()
