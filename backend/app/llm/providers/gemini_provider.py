from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from app.llm.providers.base import BaseLLMProvider
from app.config import settings

class GeminiProvider(BaseLLMProvider):
    def get_llm(self, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
        model = model_name or settings.LLM_MODEL or "gemini-1.5-pro"
        api_key = settings.GOOGLE_API_KEY
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
            
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )
