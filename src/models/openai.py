import os, getpass
from langchain_openai import ChatOpenAI
from enum import Enum

# 3/26/25 - Reference: https://platform.openai.com/docs/models
class ModelTier(Enum):
    ECONOMY = "gpt-4o-mini-2024-07-18"   # cheap option, $0.150 / 1M tokens
    ECO_REASONING = "o3-mini-2025-01-31" # latest cheap reasoning model, $1.10 / 1M tokens
    PREMIUM = "gpt-4o-2024-08-06"   # full non-reasoning model, $2.5 / 1M tokens
    PREM_REASONING = "o1-2024-12-17" # full reasoning model, o1, $15 / 1M tokens
    ULTIMATE = "gpt-4.5-preview-2025-02-27" # Most expensive, best performance, $75 / 1M tokens

# Query OpenAI
def get_openai_model(model_str = ModelTier.ECONOMY.value):
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    return ChatOpenAI(model=model_str)
    
