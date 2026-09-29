"""
LangGraph Agentic RAG Graph for Enterprise Multi-Modal Financial Intelligence.

Features:
- StateGraph with checkpointer (MemorySaver) for state persistence and time-travel in LangGraph Studio.
- RAG Retrieval Tool: Qdrant Hybrid Search (Dense Qwen 1024d + Sparse FastEmbed BM25 + RRF Fusion), 10 chunks per query.
- Multi-rewrite query execution allowing the agent to retrieve multiple angles (tables, narrative, figures).
- Grounded Synthesis Node: local Ollama gemma4:26b (num_ctx=8192) for deep long-context answer generation with page citations.
- Agent Orchestration LLM: local Ollama qwen3.5:9b for fast tool-call-capable query planning and routing.

Graph Architecture:
    START
      |
    [agent]  <-- qwen3.5:9b, only sees retrieve_chunks tool
      |
      +-- has tool_calls --> [tools]  <-- Qdrant hybrid + BGE reranker
      |                         |
      |                    retrieval_count < 2 --> [agent]  (another rewrite pass)
      |                    retrieval_count >= 2 -> [synthesize]
      |
      +-- no tool_calls  --> [synthesize]  <-- gemma4:26b, num_ctx=8192
                                 |
                               [END]

Design Decisions:
- generate_grounded_answer is NOT exposed to the agent as a tool. The agent's only
  job is retrieval planning. Synthesis is always handled by the dedicated synthesize_node,
  guaranteeing: (a) synthesis always happens, (b) no double-invocation risk,
  (c) clean separation of concerns between planner and synthesizer models.
"""

from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agent.models import get_agent_llm
from agent.tools import retrieve_chunks, generate_grounded_answer, normalize_text_input
from agent.prompt import AGENT_SYSTEM_PROMPT


# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieval_count: int


# 2. Initialize Models & Checkpointer
checkpointer = MemorySaver()
agent_llm = get_agent_llm()  # qwen3.5:9b — orchestrator only

# NOTE: generate_grounded_answer is intentionally excluded from agent_tools.
# The agent's sole responsibility is retrieval planning via retrieve_chunks.
# Synthesis is always handled by the dedicated synthesize_node (gemma4:26b).
agent_tools = [retrieve_chunks]
available_tools = {t.name: t for t in agent_tools}


# 3. Graph Node Definitions
def agent_node(state: AgentState):
    """Agent decision node: analyzes inquiry, plans query rewrites, and invokes retrieve_chunks.

    Uses qwen3.5:9b (lightweight, fast) for tool-call routing and query planning.
    The agent only has access to retrieve_chunks — it cannot call generate_grounded_answer.
    Synthesis is always performed by the dedicated synthesize_node after retrieval completes.
    """
    msgs = state["messages"]
    count = state.get("retrieval_count", 0)

    # Dynamic prompt guidance to prevent runaway retrieval loops
    if count >= 2:
        prompt_text = (
            AGENT_SYSTEM_PROMPT
            + "\n\nCRITICAL: You have already completed 2 retrieval passes. "
            "Do NOT call retrieve_chunks again. Stop now and the system will automatically synthesize the final answer."
        )
    else:
        prompt_text = AGENT_SYSTEM_PROMPT

    model = agent_llm.bind_tools(agent_tools)
    system = SystemMessage(content=prompt_text)
    response = model.invoke([system] + msgs)
    return {"messages": [response]}


def tool_node(state: AgentState):
    """Tool execution node: executes Qdrant hybrid retrieval (retrieve_chunks only)."""
    last_msg = state["messages"][-1]
    count = state.get("retrieval_count", 0)
    tool_messages = []

    for tc in getattr(last_msg, "tool_calls", []):
        t_name = tc["name"]
        t_args = tc["args"]
        t_id = tc["id"]

        if t_name in available_tools:
            tool = available_tools[t_name]
            try:
                result = tool.invoke(t_args)
            except Exception as e:
                result = f"Error executing tool '{t_name}': {e}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=t_id, name=t_name))
            if t_name == "retrieve_chunks":
                count += 1

    return {"messages": tool_messages, "retrieval_count": count}


def synthesize_node(state: AgentState):
    """Synthesis node: the single, guaranteed path for answer generation.

    Uses gemma4:26b (num_ctx=8192) via generate_grounded_answer.
    Always fires — whether the agent stopped voluntarily or hit the retrieval budget.
    Aggregates all retrieve_chunks ToolMessage outputs as the evidence context.
    """
    # Find original user query (normalize str or Studio UI multimodal block list)
    user_q = ""
    for m in state["messages"]:
        if getattr(m, "type", "") == "human":
            user_q = normalize_text_input(m.content)
            break
    if not user_q and state["messages"]:
        user_q = normalize_text_input(state["messages"][0].content)

    # Collect all retrieved evidence from retrieve_chunks tool responses
    evidence_pieces = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.name == "retrieve_chunks":
            evidence_pieces.append(normalize_text_input(m.content))

    if not evidence_pieces:
        # Edge case: agent produced no retrieval results
        return {"messages": [AIMessage(content="No evidence was retrieved. Please rephrase your question.")]}

    combined_evidence = "\n\n".join(evidence_pieces)
    try:
        grounded_ans = generate_grounded_answer.invoke({
            "question": user_q,
            "retrieved_evidence": combined_evidence
        })
    except Exception as e:
        grounded_ans = f"Error generating grounded answer: {e}"

    return {"messages": [AIMessage(content=str(grounded_ans))]}


# 4. Conditional Edge Routing
def should_continue(state: AgentState):
    """Route after agent_node.

    - Has tool_calls (retrieve_chunks) -> execute them in tool_node
    - No tool_calls (agent is done retrieving) -> go to synthesize_node
      (synthesis ALWAYS happens — no path to END without synthesis)
    """
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    return "synthesize"


def after_tools(state: AgentState):
    """Route after tool_node.

    - retrieval_count >= 2 -> synthesize (budget exhausted)
    - otherwise -> back to agent for another retrieval rewrite pass
    """
    if state.get("retrieval_count", 0) >= 2:
        return "synthesize"
    return "agent"


# 5. Build and Compile Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("synthesize", synthesize_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "synthesize"])
workflow.add_conditional_edges("tools", after_tools, ["agent", "synthesize"])
workflow.add_edge("synthesize", END)

# Export compiled graph for LangGraph Studio
graph = workflow.compile()
