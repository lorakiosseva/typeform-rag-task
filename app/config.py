import os
from dotenv import load_dotenv

# Load environment variables from .env (for local dev).
# In Docker/production, env vars can be passed directly and will override.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "typeform-helpcenter")
APP_ENV = os.getenv("APP_ENV", "local")


def validate_config() -> None:
    """Validate that required environment variables are present."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_str}")


# Run validation on import so failures are fast & obvious.
validate_config()