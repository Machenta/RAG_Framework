import asyncio
from typing import Any, Dict
from ..base import DataFetcher
from rag_shared.utils.config import Config
from rag_shared.utils.retrieval import Retrieval
from rag_shared.core.fetchers.registry import get_processor

class AzureSearchFetcher(DataFetcher):
    """
    Adapts your existing Retrieval.search(...) method
    to the async DataFetcher.fetch(...) interface.
    """
    def __init__(self, config: Config):
        self._retrieval = Retrieval(config=config)
        az_search_cfg = config.app.fetchers.AzureSearchFetcher
        self.default_proc = az_search_cfg.processor


    @classmethod
    def build_args(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the kwargs for fetch() with defaults:

        Required:
          - context['user_question'] → query
          - context['filter']        → filter

        Defaults:
          top_k                = 5
          skip                 = 0
          include_total_count  = False
          facets               = None
          highlight_fields     = None
          search_fields        = None
          select_fields        = None
          semantic             = False
          semantic_config      = None
          vector_search        = False
          hybrid               = None
          vector_field         = "contentVector"
        """
        q = context.get("user_question")
        f = context.get("filter")
        if not q or not f:
            # missing the absolutely required pieces → skip this fetcher
            return {}

        return {
            "query":                q,
            "filter":               f,
            "top_k":                context.get("top_k", 5),
            "skip":                 context.get("skip", 0),
            "include_total_count":  context.get("include_total_count", False),
            "facets":               context.get("facets", None),
            "highlight_fields":     context.get("highlight_fields", None),
            "search_fields":        context.get("search_fields", None),
            "select_fields":        context.get("select_fields", None),
            "semantic":             context.get("semantic", False),
            "semantic_config":      context.get("semantic_config", None),
            "vector_search":        context.get("vector_search", False),
            "hybrid":               context.get("hybrid", None),
            "vector_field":         context.get("vector_field", "contentVector"),
        }

    async def fetch(self, **kwargs) -> Dict[str, Any]:
        # Extract parameters from kwargs with defaults
        query = kwargs.get("query")
        if query is None:
            raise ValueError("The 'query' parameter is required and must be a string.")
        top_k = kwargs.get("top_k", 5)
        skip = kwargs.get("skip", 0)
        filter = kwargs.get("filter", None)
        order_by = kwargs.get("order_by", None)
        facets = kwargs.get("facets", None)
        highlight_fields = kwargs.get("highlight_fields", None)
        search_fields = kwargs.get("search_fields", None)
        select_fields = kwargs.get("select_fields", None)
        include_total_count = kwargs.get("include_total_count", False)
        semantic = kwargs.get("semantic", False)
        semantic_config = kwargs.get("semantic_config", None)
        vector_search = kwargs.get("vector_search", False)
        hybrid = kwargs.get("hybrid", None)
        vector_field = kwargs.get("vector_field", "contentVector")

        raw = await asyncio.to_thread(
            self._retrieval.search,
            query,
            top_k=top_k,
            skip=skip,
            filter=filter,
            order_by=order_by,
            facets=facets,
            highlight_fields=highlight_fields,
            search_fields=search_fields,
            select_fields=select_fields,
            include_total_count=include_total_count,
            semantic=semantic,
            semantic_config=semantic_config,
            vector_search=vector_search,
            hybrid=hybrid,
            vector_field=vector_field,
        )

        # 1) Check if we got any results
        if not raw or not isinstance(raw, dict):
            raise ValueError(
                f"AzureSearchFetcher returned invalid data: {raw!r}. "
                "Expected a non-empty dictionary."
            )
        if "results" not in raw or not isinstance(raw["results"], list):
            raise ValueError(
                f"AzureSearchFetcher returned invalid data: {raw!r}. "
                "Expected a dictionary with a 'results' key containing a list."
            )
        if not raw["results"]:
            raise ValueError(
                f"AzureSearchFetcher returned no results for query: {query!r}. "
                "Check your query parameters and data."
            )

        # 2) Determine which processor to use
        proc_name = kwargs.get("processor") or self.default_proc
        print(f"Using processor: {proc_name}")
        # lookup the function in your registry
        proc = get_processor(self.__class__.__name__, proc_name)
        if proc is None:
            raise ValueError(
                f"No processor registered for "
                f"{self.__class__.__name__!r} + {proc_name!r}"
            )

        # 3) Apply it and return
        return proc(raw)  
    

if __name__ == "__main__":

    cfg = Config(key_vault_name="RecoveredSpacesKV", config_folder="resources/configs", config_filename="recovered_config.yml")

    async def main():
        fetcher = AzureSearchFetcher(config=cfg)

        filter_str = "speaker eq 'David' and timestamp ge 0 and timestamp le 1000"
        query = "What did David say about Anastasiya?"

        response = await fetcher.fetch(
            query=query,
            filter=filter_str,
            top_k=5,
            include_total_count=True,
            facets=["speaker,count:5", "topic"],
            highlight_fields=["text"],
            select_fields=[
                "id","filename","block_id","chunk_index","part","speaker",
                "timestamp","tokens","video_url","keyword","topic","text"
            ],
            vector_search=True
        )

        print(f"Search results for query '{query}' with filter '{filter_str}':")
        print("=== Response Metadata ===")
        print(f"Total count:       {response.get('total_count')}")
        print("Facets:")
        for field, buckets in response.get("facets", {}).items():
            print(f"  {field}:")
            for b in buckets:
                print(f"    {b['value']} → {b['count']}")
        print()
        print("=== Documents ===")
        for doc in response.get("results", []):
            print(f"ID:           {doc['id']}")
            print(f"Filename:     {doc.get('filename')}")
            print(f"Block/Chunk:  {doc['block_id']} (#{doc['chunk_index']})")
            print(f"Part:         {doc['part']}")
            print(f"Speaker:      {doc['speaker']}")
            print(f"Timestamp:    {doc['timestamp']}s")
            print(f"Tokens:       {doc['tokens']}")
            if doc.get("video_url"):
                print(f"Video URL:    {doc['video_url']}")
            if doc.get("keyword"):
                print(f"Keyword:      {doc['keyword']}")
            if doc.get("topic"):
                print(f"Topic:        {doc['topic']}")
            highlights = doc.get("@search.highlights")
            if highlights:
                print("Highlights:")
                for field, snippets in highlights.items():
                    for snippet in snippets:
                        print("  →", snippet)
            print("Text:")
            print(doc["text"])
            print(f"Score:        {doc['_score']:.3f}")
            print("-" * 60)

    asyncio.run(main())