# app/core/fetchers/postgres_fetcher.py

import asyncio
from typing import Any, Dict, Optional, Union, List
import asyncpg
from rag_shared.core.fetchers.base import DataFetcher

class PostgresFetcher(DataFetcher):
    def __init__(self, dsn: str):
        self._dsn  = dsn
        self._pool = None

    async def fetch(self, **kwargs) -> Dict[str, Any]:
        # 1) Extract our expected args
        sql    = kwargs.get("sql")
        params = kwargs.get("params", None)

        if not sql or not isinstance(sql, str):
            raise ValueError("`sql` must be provided as a string.")

        # 2) Offload to thread or run in asyncpg directly
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if isinstance(params, dict):
                records = await conn.fetch(sql, **params)
            elif isinstance(params, list):
                records = await conn.fetch(sql, *params)
            else:
                records = await conn.fetch(sql)

        rows = [dict(r) for r in records]
        return {
            "source": "postgres",
            "sql": sql,
            "rows": rows
        }

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(dsn=self._dsn)
        return self._pool

if __name__ == "__main__":
    import os
    import asyncio
    from rag_shared.utils.config import Config

    # 1) Load your Postgres DSN from config or env
    cfg = Config()  
    # expects cfg.postgres_dsn or you can read from env directly:
    dsn = getattr(cfg, "postgres_dsn", os.getenv("POSTGRES_DSN"))
    if not dsn:
        raise RuntimeError("POSTGRES_DSN not set in Config or environment")

    # 2) Instantiate the fetcher
    fetcher = PostgresFetcher(dsn=dsn)

    async def main():
        # 3) Define a simple test query
        print("📡 Running test query: SELECT 1 as one;")
        resp = await fetcher.fetch(sql="SELECT 1 AS one;")
        print("✅ Response:")
        print(resp)

        # 4) Another sample: list tables in public schema
        print("\n📡 Running test query: SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
        resp2 = await fetcher.fetch(
            sql="SELECT table_name FROM information_schema.tables WHERE table_schema=$1;",
            params=["public"]
        )
        print("✅ Tables in public schema:")
        for row in resp2["rows"]:
            print(" -", row["table_name"])

    asyncio.run(main())
