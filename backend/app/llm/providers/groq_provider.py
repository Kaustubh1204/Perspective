from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from app.llm.providers.base import BaseLLMProvider
from app.config import settings

class GroqProvider(BaseLLMProvider):
    def get_llm(self, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
        model = model_name or settings.LLM_MODEL or "llama-3.3-70b-versatile"
        api_key = settings.GROQ_API_KEY
        
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
            
        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )
