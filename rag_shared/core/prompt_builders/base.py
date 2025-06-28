from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List

class PromptBuilder(ABC):
    @abstractmethod
    def build(
        self,
        fetched: Dict[str, Any],
        user_question: str
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Return either:
         - a single prompt string, or
         - a chat‐style list of {"role":..., "content":...} dicts.
        """
        ...
