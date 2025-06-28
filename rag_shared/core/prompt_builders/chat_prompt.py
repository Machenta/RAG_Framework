from .base import PromptBuilder

class ChatPromptBuilder(PromptBuilder):
    def build(self, fetched, user_question):
        snippets = [r["text"] for r in fetched["AzureSearchFetcher"]["results"][:3]]
        return [
            {"role":"system","content":"You are an expert."},
            {"role":"user","content":"\n\n".join(snippets)},
            {"role":"user","content":f"Answer: {user_question}"}
        ]
