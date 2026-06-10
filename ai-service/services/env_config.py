import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent


def load_app_env() -> None:
    """Load env files; chatBot/.env overrides ai-service/.env."""
    load_dotenv(APP_DIR / ".env")
    load_dotenv(APP_DIR / "chatBot" / ".env", override=True)


def get_openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
