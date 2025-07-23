from .base import PromptBuilder
from typing import Any, Dict, List

class ChatPromptBuilder(PromptBuilder):
    def build(self, fetched: Dict[str, Any], user_question: str) -> List[Dict[str, str]]:
        # Build content with metadata for better source attribution
        content_with_sources = []
        
        # Get the first 3 results with metadata
        results = fetched.get("AzureSearchFetcher", {}).get("results", [])[:3]
        
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            source = result.get("filename", "Unknown Document")
            speaker = result.get("speaker", "Unknown Speaker")
            topic = result.get("topic", "General")
            qa_type = result.get("questionoranswer", "content")
            timestamp = result.get("timestamp", "")
            
            source_info = f"Source {i}: {source}"
            if speaker and speaker != "Unknown Speaker":
                source_info += f" - {speaker}"
            if topic and topic != "General":
                source_info += f" ({topic})"
            if timestamp:
                source_info += f" [Time: {timestamp}]"
            if qa_type:
                source_info += f" [{qa_type}]"
            
            content_with_sources.append(f"{source_info}\n{text}")
        
        context = "\n\n---\n\n".join(content_with_sources)
        
        return [
            {
                "role": "system", 
                "content": "You are an expert assistant. When answering questions, always cite your sources using the format '[Source: Document Name (Page X)]' when referencing specific information."
            },
            {
                "role": "user", 
                "content": f"Context Information:\n\n{context}"
            },
            {
                "role": "user", 
                "content": f"Question: {user_question}\n\nPlease answer based on the provided context and cite your sources appropriately."
            }
        ]
