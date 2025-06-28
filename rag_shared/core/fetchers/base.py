from abc import ABC, abstractmethod
from typing import Any, Dict

class DataFetcher(ABC):
    """Common interface for every data‑source implementation."""

    @abstractmethod
    async def fetch(self, **kwargs) -> Dict[str, Any]:
        """Return JSON‑serialisable dict with the retrieved data."""


    def process(self, data: Dict[str,Any]) -> Dict[str,Any]:
        """
        Optional post‐processing hook. By default, returns data unchanged.
        Subclasses can override to compute averages, filter, reshape, etc.
        """
        return data