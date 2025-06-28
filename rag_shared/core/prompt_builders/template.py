from typing import Any, Dict, Union, List
from jinja2 import Template
from .base import PromptBuilder
import json

class TemplatePromptBuilder(PromptBuilder):
    def __init__(self, template_str: str):
        self.template = Template(template_str)

    def build(
        self,
        fetched: Dict[str, Any],
        user_question: str
    ) -> Union[str, List[Dict[str, str]]]:
        rendered = self.template.render(data=fetched, question=user_question)
        # if the template rendered JSON array → use as messages
        try:
            obj = json.loads(rendered)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        return rendered
