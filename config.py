import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment once for the app
load_dotenv()

# Model and API configuration
DEFAULT_INTENT_MODEL = os.getenv("AI_INTENT_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
DEFAULT_RAG_MODEL = os.getenv("AI_RAG_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or os.getenv("MODEL_API_BASE")


def build_openai_client() -> Optional[OpenAI]:
    """Create a shared OpenAI client if API key is configured."""
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
