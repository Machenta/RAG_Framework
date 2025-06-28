
import asyncio
from typing import Any, Dict, Optional
import httpx
from rag_shared.core.fetchers.base import DataFetcher
from rag_shared.utils.config import Config
from rag_shared.core.fetchers.registry import get_processor

class RestAPIFetcher(DataFetcher):
    def __init__(self, base_url: str, token: str, config: Config):
        fetch_cfg = config.app.fetchers.RestAPIFetcher
        self.base_url     = fetch_cfg.params.base_url
        self.token        = fetch_cfg.params.token or ""
        self.default_proc = fetch_cfg.processor
        self.config       = config

    async def fetch(self, **kwargs) -> Dict[str, Any]:
            route = kwargs.get("route")
            if not route:
                raise ValueError("`route` is required (e.g. 'posts' or 'users/1').")
            params = kwargs.get("params", {})

            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url.rstrip('/')}/{route.lstrip('/')}",
                    params=params,
                    headers=headers,
                    timeout=10
                )
                resp.raise_for_status()
                raw = resp.json()

            # pick processor: override in fetch_args, else YAML default
            proc_name = kwargs.get("processor") or self.default_proc
            print(f"Using processor: {proc_name}")
            if not proc_name:
                raise ValueError("No processor specified for RestAPIFetcher.")

            # lookup & apply
            print("All processors:", self.config.app.fetchers.RestAPIFetcher.processor)
            processor_fn = get_processor("RestAPIFetcher", proc_name)
            print("Raw data:", raw)
            processed = processor_fn(raw)

            return {
                "source": "rest_api",
                "url":    str(resp.url),
                "data":   processed
            }


if __name__ == "__main__":
    fetcher = RestAPIFetcher("https://jsonplaceholder.typicode.com", 
                             "", 
                             config=Config(key_vault_name="RecoveredSpacesKV", config_path="environment_config.yml"))

    async def main():
        print("Fetching https://jsonplaceholder.typicode.com/posts?userId=1\n")
        resp = await fetcher.fetch(route="posts", params={"userId": 1}, processor="flatten_user_dict")

        print("URL:", resp["url"])
        print("Number of items:", len(resp["data"]))
        print("Sample item:", resp["data"][0])

    asyncio.run(main())