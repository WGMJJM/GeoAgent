"""模型 Provider 实现。"""

from .openai_compatible import OpenAICompatibleAdapter
from .openai_responses import OpenAIResponsesAdapter

__all__ = ["OpenAICompatibleAdapter", "OpenAIResponsesAdapter"]

