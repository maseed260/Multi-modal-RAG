import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

# Model specifications
PREFERRED_AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.1-8b-instant")
FALLBACK_AGENT_MODEL = "openai/gpt-oss-20b"

PREFERRED_SYNTHESIS_MODEL = os.getenv("GROQ_SYNTHESIS_MODEL", "llama-3.3-70b-versatile")
FALLBACK_SYNTHESIS_MODEL = "openai/gpt-oss-120b"


def get_agent_llm() -> ChatGroq:
    """Return the LLM for agent decision making and tool orchestrating."""
    try:
        llm = ChatGroq(model=PREFERRED_AGENT_MODEL, api_key=groq_api_key, temperature=0.1)
        # Test invocation to check if model exists
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"[*] Note: '{PREFERRED_AGENT_MODEL}' not available ({e}). Using '{FALLBACK_AGENT_MODEL}'.")
        return ChatGroq(model=FALLBACK_AGENT_MODEL, api_key=groq_api_key, temperature=0.1)


def get_synthesis_llm() -> ChatGroq:
    """Return the high-capacity LLM for grounded answer generation."""
    try:
        llm = ChatGroq(model=PREFERRED_SYNTHESIS_MODEL, api_key=groq_api_key, temperature=0.2)
        llm.invoke("ping")
        return llm
    except Exception as e:
        print(f"[*] Note: '{PREFERRED_SYNTHESIS_MODEL}' not available ({e}). Using '{FALLBACK_SYNTHESIS_MODEL}'.")
        return ChatGroq(model=FALLBACK_SYNTHESIS_MODEL, api_key=groq_api_key, temperature=0.2)
