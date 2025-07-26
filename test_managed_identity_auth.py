#!/usr/bin/env python3
"""
Test script to verify System-Assigned Managed Identity authentication for Azure services,
with detailed error reporting at each phase and optional API-key fallback.
"""

import os
import logging
import json
import base64

from rag_shared.utils.config import Config
from rag_shared.utils.retrieval import Retrieval

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ─── Enable verbose HTTP logging ───────────────────────────────────────────────
os.environ["AZURE_CORE_HTTP_LOGGING_ENABLE"] = "ALL"
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

# ─── Optional: provide an API key to bypass Managed Identity (empty = use AD) ──
# You can also set this via env var AZURE_SEARCH_API_KEY
#PMH9n8Cs7iaIGBU113rNxbl5JpA4P9O5vr5LSodW5eAzSeBiTDkp
api_key = os.getenv("AZURE_SEARCH_API_KEY", "")  # e.g. "YOUR-QUERY-KEY-HERE"

def _who_am_i_for_search():
    """Helper to print the OIDC token claims for the Search audience."""
    cred = DefaultAzureCredential()
    tok = cred.get_token("https://search.azure.com/.default").token
    payload_b64 = tok.split('.')[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + '==='))  # noqa
    logging.info(
        "🔐 Identity claims → oid=%s, appid=%s, tid=%s",
        payload.get("oid"), payload.get("appid"), payload.get("tid"),
    )
    return payload

def _dump_http_error(err: HttpResponseError, context: str):
    """Print status, and parse JSON body for code/message if present."""
    print(f"❌ HttpResponseError in {context}:")
    print(f"   Status code : {getattr(err, 'status_code', 'N/A')}")
    print(f"   Message     : {getattr(err, 'message', str(err))}")
    try:
        if err.response is not None:
            body = json.loads(err.response.text())
            err_code = body.get("error", {}).get("code")
            err_msg  = body.get("error", {}).get("message")
            if err_code or err_msg:
                print("   Parsed error payload:")
                if err_code:
                    print(f"     code   : {err_code}")
                if err_msg:
                    print(f"     message: {err_msg}")
    except Exception:
        pass
    try:
        if err.response is not None:
            text = err.response.text()
            if text:
                print("   Raw response text:")
                print(text)
    except Exception:
        pass

def test_managed_identity_auth():
    """Test that managed identity authentication is being used correctly."""
    print("🧪 Testing System-Assigned Managed Identity Authentication")
    print("=" * 60)

    # 1) Load config & init Retrieval
    try:
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        print("✅ Config loaded successfully")
        print(f"   App name: {config.app.name}")
        print("\n🔧 Configuration Check:")
        print(f"   AI Search MI:       {getattr(getattr(config.app, 'ai_search', None), 'use_managed_identity', 'N/A')}")
        print(f"   LLM MI:             {getattr(getattr(config.app, 'llm', None), 'use_managed_identity', 'N/A')}")
        emb = getattr(getattr(config.app, 'ai_search', None), 'index', None)
        emb = getattr(emb, 'embedding', None)
        print(f"   Embedding MI:       {getattr(emb, 'use_managed_identity', 'N/A')}")
    except Exception as e:
        print(f"❌ Failed loading config: {e}")
        return

    print("\n🔍 Testing Retrieval Class Initialization:")
    try:
        retrieval = Retrieval(config=config)
        print("✅ Retrieval class initialized successfully")
        print(f"   Index name: {retrieval._index_cfg.name}")
    except Exception as e:
        print(f"❌ Retrieval init failed: {e}")
        return

    # 2) Decode which identity is being used (only for AD path)
    if not api_key:
        _who_am_i_for_search()

    # Manual SearchClient tests
    endpoint   = "https://ragtests.search.windows.net"
    index_name = "unified_text_index"

    # ─── A) Construct SearchClient ─────────────────────────────────────────────
    try:
        if api_key:
            cred   = AzureKeyCredential(api_key)
            logging.info("🔑 Using API key authentication for SearchClient")
        else:
            cred   = DefaultAzureCredential()
            logging.info("🔐 Using DefaultAzureCredential for SearchClient")
        client = SearchClient(endpoint=endpoint, index_name=index_name, credential=cred)
        logging.info("✅ SearchClient constructed successfully")
    except Exception as e:
        logging.error("❌ Failed to construct SearchClient:")
        logging.exception(e)
        return

    # ─── B) Test get_index (management‑plane) ──────────────────────────────────
    try:
        if api_key:
            idx_cred = AzureKeyCredential(api_key)
            idx_client = SearchIndexClient(endpoint=endpoint, credential=idx_cred)
            logging.info("🔑 Using API key for SearchIndexClient")
        else:
            idx_client = SearchIndexClient(endpoint=endpoint, credential=cred)
            logging.info("🔐 Using DefaultAzureCredential for SearchIndexClient")

        idx = idx_client.get_index(index_name)
        logging.info("✅ get_index succeeded; index has %d fields", len(idx.fields))
    except HttpResponseError as err:
        _dump_http_error(err, "get_index()")
        return
    except Exception as e:
        logging.error("❌ Unexpected exception in get_index():")
        logging.exception(e)
        return

    # ─── C) Test get_document_count (data‑plane) ───────────────────────────────
    try:
        count = client.get_document_count()
        logging.info("🔢 Document count: %d", count)
    except HttpResponseError as err:
        _dump_http_error(err, "get_document_count()")
        return
    except Exception as e:
        logging.error("❌ Unexpected exception in get_document_count():")
        logging.exception(e)
        return

    # ─── D) Test a simple search ───────────────────────────────────────────────
    try:
        docs = list(client.search(search_text="test", top=1))
        logging.info("🔍 Search returned %d document(s)", len(docs))
        for d in docs:
            logging.info(" • %s", d)
    except HttpResponseError as err:
        _dump_http_error(err, "search()")
    except Exception as e:
        logging.error("❌ Unexpected exception in search():")
        logging.exception(e)

    print("\n🎉 Authentication and search tests completed!")

if __name__ == "__main__":
    test_managed_identity_auth()
