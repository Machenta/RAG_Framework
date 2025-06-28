from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMModel(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response. Either:
        - pass a single `prompt` string, or
        - pass a list of chat `messages` dicts [{"role":..., "content":...}, ...].
        """
        ...
