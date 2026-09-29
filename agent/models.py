import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

# Automatically configure LangSmith tracing if API key is detected
langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "multi-modal-rag")

# ── Two-model Ollama configuration ────────────────────────────────────────────
#
#  Role                | Model          | Purpose
#  --------------------|----------------|------------------------------------
#  Agent Orchestrator  | qwen3.5:9b     | Tool-call routing, query planning
#  Answer Synthesizer  | gemma4:26b     | Long-context grounded answer gen
#
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Orchestrator: Qwen 3.5 9B — fast, efficient tool-calling for agent decisions
OLLAMA_AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "qwen3.5:9b")

# Synthesizer: Gemma 4 26B — large context, deep reasoning for executive-grade answers
OLLAMA_SYNTHESIS_MODEL = os.getenv("OLLAMA_SYNTHESIS_MODEL", "gemma4:26b")

# ── Optional Groq fallback configuration ──────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")
GROQ_SYNTHESIS_MODEL = os.getenv("GROQ_SYNTHESIS_MODEL", "openai/gpt-oss-120b")


def get_agent_llm():
    """Return the LLM for agent decision making and tool orchestration.

    Default: local Ollama qwen3.5:9b — fast, tool-call-capable, zero API rate limits.
    Override via OLLAMA_AGENT_MODEL env var.
    """
    if LLM_PROVIDER == "ollama":
        print(f"[*] Agent Orchestrator  : Ollama '{OLLAMA_AGENT_MODEL}' @ {OLLAMA_URL}")
        return ChatOllama(
            model=OLLAMA_AGENT_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.1,
        )

    # Fallback to Groq API if explicitly selected
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER='groq'.")
    print(f"[*] Agent Orchestrator  : Groq '{GROQ_AGENT_MODEL}'")
    return ChatGroq(model=GROQ_AGENT_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def get_synthesis_llm():
    """Return the LLM for grounded answer synthesis.

    Default: local Ollama gemma4:26b — high-capacity, 8K context for long-context
    multi-modal reconciliation and executive-grade answer generation.
    Override via OLLAMA_SYNTHESIS_MODEL env var.
    """
    if LLM_PROVIDER == "ollama":
        print(f"[*] Answer Synthesizer  : Ollama '{OLLAMA_SYNTHESIS_MODEL}' @ {OLLAMA_URL}")
        return ChatOllama(
            model=OLLAMA_SYNTHESIS_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.2,
            # Expand context window beyond the default 2048 tokens to handle
            # long retrieved evidence passages from multi-chunk queries
            num_ctx=8192,
        )

    # Fallback to Groq API if explicitly selected
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER='groq'.")
    print(f"[*] Answer Synthesizer  : Groq '{GROQ_SYNTHESIS_MODEL}'")
    return ChatGroq(model=GROQ_SYNTHESIS_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
