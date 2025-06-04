import os
from dotenv import load_dotenv

def get_openai_key() -> str:
    """
    Reads the OpenAI API key from .env or environment variables.

    Returns:
        str: OpenAI API key
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY not found! Please set in .env file.")
    return api_key
