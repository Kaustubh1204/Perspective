from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from app.llm.providers.base import BaseLLMProvider
from app.config import settings

class OpenAIProvider(BaseLLMProvider):
    def get_llm(self, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
        model = model_name or settings.LLM_MODEL or "gpt-4o"
        api_key = settings.OPENAI_API_KEY
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
            
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )
