"""Configuration settings for the ServiceNow Ticket Analysis Agent."""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API Configuration - Organization Wrapper
# =============================================================================

# OAuth2 Authentication Settings
AUTH_URL = os.getenv("AUTH_URL", "https://your-org-auth.example.com/oauth2/token")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")

# GPT Wrapper API Settings
# URL format: https://your-org/{model}/chat/completions?api-version={version}
GPT_WRAPPER_BASE_URL = os.getenv("GPT_WRAPPER_BASE_URL", "https://your-org-gpt-wrapper.example.com")
GPT_API_VERSION = os.getenv("GPT_API_VERSION", "2024-02-15-preview")

# Optional: Direct OpenAI API Key (if wrapper not configured)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Use wrapper API if credentials are provided, otherwise fall back to direct OpenAI
USE_WRAPPER_API = bool(CLIENT_ID and CLIENT_SECRET and GPT_WRAPPER_BASE_URL)

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")  # Can also use "gpt-4o-mini"
MAX_TOKENS = 4096


def get_gpt_wrapper_url(model_name: str = None) -> str:
    """Build the full GPT wrapper URL with model embedded.

    Format: {base_url}/{model}/chat/completions?api-version={version}
    """
    model = model_name or MODEL_NAME
    return f"{GPT_WRAPPER_BASE_URL}/{model}/chat/completions?api-version={GPT_API_VERSION}"

# =============================================================================
# Excel Column Configuration (matches ServiceNow export)
# =============================================================================
COLUMN_MAPPING = {
    "row_number": "#",
    "incident_number": "Incident Number",
    "short_description": "Short Description",
    "description": "Description",
    "solution": "Solution",
    "priority": "Priority",
    "category": "Category",  # New column to be added
}

# =============================================================================
# Categorization Settings
# =============================================================================
BATCH_SIZE = 20  # Number of tickets to analyze per API call
SIMILARITY_THRESHOLD = 0.7  # Threshold for considering tickets similar
MAX_CATEGORIES = 25  # Maximum number of categories to create

# =============================================================================
# Output Settings
# =============================================================================
OUTPUT_SUFFIX = "_categorized"  # Suffix for output file
