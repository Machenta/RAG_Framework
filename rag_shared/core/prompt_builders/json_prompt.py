import json
from .base import PromptBuilder

class JSONPromptBuilder(PromptBuilder):
    def build(self, fetched, user_question):
        ctx = json.dumps(fetched, indent=2)
        return (
            "Context:\n" + ctx + "\n\n"
            f"Question: {user_question}\nAnswer:"
        )
