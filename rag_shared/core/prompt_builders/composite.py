from typing import Any, Dict, List, Union
from .base import PromptBuilder

class CompositePromptBuilder(PromptBuilder):
    def __init__(self,
                 builders: List[PromptBuilder],
                 sep: str = "\n\n"):
        self.builders = builders
        self.sep      = sep

    def build(self, fetched, user_question) -> Union[str, List[Dict[str,str]]]:
        parts = [b.build(fetched, user_question) for b in self.builders]
        if all(isinstance(p, str) for p in parts):
            return self.sep.join(p for p in parts if isinstance(p, str))
        merged: List[Dict[str,str]] = []
        for p in parts:
            if isinstance(p, str):
                merged.append({"role":"user","content":p})
            else:
                merged.extend(p)
        return merged
