import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

# Primary local configuration: Ollama with Qwen 9B
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# Optional Groq fallback configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "openai/gpt-oss-20b")
GROQ_SYNTHESIS_MODEL = os.getenv("GROQ_SYNTHESIS_MODEL", "openai/gpt-oss-120b")


def get_agent_llm():
    """Return the LLM for agent decision making and tool orchestration.
    
    Defaults to local Ollama qwen3.5:9b for zero-rate-limit local execution.
    """
    if LLM_PROVIDER == "ollama":
        print(f"[*] Using local Ollama model '{OLLAMA_MODEL}' at {OLLAMA_URL} for Agent decisions.")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.1
        )
    
    # Fallback to Groq API if explicitly selected
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER='groq'.")
    print(f"[*] Using Groq model '{GROQ_AGENT_MODEL}' for Agent decisions.")
    return ChatGroq(model=GROQ_AGENT_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


def get_synthesis_llm():
    """Return the LLM for grounded answer synthesis.
    
    Defaults to local Ollama qwen3.5:9b for 100% local privacy and execution.
    """
    if LLM_PROVIDER == "ollama":
        print(f"[*] Using local Ollama model '{OLLAMA_MODEL}' at {OLLAMA_URL} for Answer Synthesis.")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0.2
        )
    
    # Fallback to Groq API if explicitly selected
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER='groq'.")
    print(f"[*] Using Groq model '{GROQ_SYNTHESIS_MODEL}' for Answer Synthesis.")
    return ChatGroq(model=GROQ_SYNTHESIS_MODEL, api_key=GROQ_API_KEY, temperature=0.2)
