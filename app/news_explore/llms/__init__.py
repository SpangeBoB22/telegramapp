from .ai_client import OpenAIConnector
from .gemini_client import GeminiConnector
from ...core.models import ModelNames

MODELS = {ModelNames.GEMINI: GeminiConnector,
          ModelNames.GPT4: OpenAIConnector}

MODELS_REQUEST_LIMITS = {ModelNames.GEMINI: 14,
                         ModelNames.GPT4: 50}
