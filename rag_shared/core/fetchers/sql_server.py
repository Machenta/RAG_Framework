# app/core/fetchers/sql_server.py

import asyncio
from typing import Any, Dict, List, Optional, Union

import pyodbc
from azure.identity import DefaultAzureCredential
from rag_shared.core.fetchers.base import DataFetcher

# ODBC constant for passing an Access Token (1256)  
# If your pyodbc/driver does not export it, define it here:
SQL_COPT_SS_ACCESS_TOKEN = 1256

class SQLServerFetcher(DataFetcher):
    """
    Fetch data from Azure SQL using Managed Identity.
    
    Config must provide:
      - sql_server:   e.g. "<your-server>.database.windows.net"
      - sql_database: your database name
      - (optional) driver name
    """
    def __init__(self,
                 server: str,
                 database: str,
                 driver: str = "{ODBC Driver 18 for SQL Server}"):
        self.server   = server
        self.database = database
        self.driver   = driver
        self.credential = DefaultAzureCredential()

    @classmethod
    def build_args(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        sql = context.get("sql")
        if not sql:
            return {}
        return {
            "sql":    sql,
            "params": context.get("params", None)
        }

    async def fetch(self, **kwargs) -> Dict[str, Any]:
        sql    = kwargs.get("sql")
        params = kwargs.get("params", None)

        if not isinstance(sql, str):
            raise ValueError("`sql` must be provided as a string.")

        # run the blocking fetch in a thread
        return await asyncio.to_thread(self._fetch_sync, sql, params)

    def _fetch_sync(
        self,
        sql: str,
        params: Optional[Union[List[Any], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # 1) Acquire an Azure AD access token for SQL DB
        token = self.credential.get_token("https://database.windows.net/.default").token
        token_bytes = token.encode("utf-16-le")

        # 2) Build ODBC connection string (no credentials needed)
        conn_str = (
            f"Driver={self.driver};"
            f"Server={self.server};"
            f"Database={self.database};"
            "Encrypt=yes;TrustServerCertificate=no;"
        )

        # 3) Connect, passing the token as a connection attribute
        conn = pyodbc.connect(conn_str,
                              attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_bytes})
        cur = conn.cursor()

        # 4) Execute query
        if params is not None:
            cur.execute(sql, params)
        else:
            cur.execute(sql)

        # 5) Fetch and map to dict
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        # 6) Clean up
        cur.close()
        conn.close()

        return {
            "source": "sql_server",
            "sql": sql,
            "rows": rows
        }


# if __name__ == "__main__":
#     import os
#     import asyncio
#     from rag_shared.utils.config import Config

#     cfg = Config()

#     fetcher = SQLServerFetcher(
#         server   = cfg.sql_server,
#         database = cfg.sql_database
#     )

#     async def main():
#         sample_sql = "SELECT TOP 5 * FROM YourTable;"
#         print(f"📡 Running SQL: {sample_sql}\n")
#         ctx = {"sql": sample_sql}
#         response = await fetcher.fetch(**SQLServerFetcher.build_args(ctx))

#         print("✅ Fetched rows:")
#         for row in response["rows"]:
#             print(row)

#     asyncio.run(main())
