from typing import Any
from rag_shared.core.fetchers.registry import register_processor

@register_processor("AzureSearchFetcher", "noop")
def noop_processor(raw: Any) -> Any:
    """
    A no‐op processor: returns the search output unchanged.
    """
    return raw