"""
Interactive Test Runner for LangGraph Multi-Modal Financial RAG Agent.
Demonstrates:
- Multi-query iterative retrieval using Qdrant Hybrid Search (Dense Qwen 1024d + Sparse BM25 + RRF)
- Grounded answer generation via Groq synthesis tool
- MemorySaver checkpointer maintaining conversation thread state
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.graph import graph


import time

def run_test(question: str):
    print("=" * 70)
    print(f"USER QUESTION: {question}")
    print("=" * 70)

    # Use unique thread_id for checkpointer state tracking
    session_id = f"session-{int(time.time())}"
    config = {"configurable": {"thread_id": session_id}}

    inputs = {"messages": [("user", question)]}

    for event in graph.stream(inputs, config=config, stream_mode="values"):
        last_message = event["messages"][-1]
        
        # Format message output
        role = getattr(last_message, "type", "message")
        if role == "ai":
            tool_calls = getattr(last_message, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get("name")
                    tool_args = tc.get("args")
                    print(f"\n[*] [AGENT ACTION] Calling Tool '{tool_name}' with args:")
                    for k, v in tool_args.items():
                        # Truncate large evidence arguments for console readability
                        val_str = str(v)
                        if len(val_str) > 160:
                            val_str = val_str[:160] + "... [truncated]"
                        print(f"   - {k}: {val_str}")
            elif last_message.content:
                print(f"\n[+] [AGENT FINAL ANSWER]:\n{last_message.content}\n")
        elif role == "tool":
            tool_name = getattr(last_message, "name", "tool")
            content_preview = str(last_message.content)[:200].replace("\n", " ")
            print(f"[-] [TOOL RESULT '{tool_name}'] Received {len(str(last_message.content))} chars: {content_preview}...")

    print("=" * 70)
    print("[OK] Run Complete.")
    print("=" * 70)


if __name__ == "__main__":
    test_q = (
        "What was the impact of the Apple Card transaction on JPMorgan Chase's 2025 "
        "financial results, specifically regarding credit loss provisions and capital ratios?"
    )
    if len(sys.argv) > 1:
        test_q = " ".join(sys.argv[1:])
    run_test(test_q)
