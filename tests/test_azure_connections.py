import pytest
import os
from rag_shared.utils.config import Config, SingletonMeta

# Import fetchers/models
from rag_shared.core.models.azure_openai import AzureOpenAIModel
from rag_shared.core.fetchers.azure_search.azure_search import AzureSearchFetcher
from rag_shared.core.fetchers.sql_server import SQLServerFetcher
from rag_shared.core.fetchers.postgres import PostgresFetcher
from rag_shared.core.fetchers.rest_api.rest_api import RestAPIFetcher
# Blob Storage fetcher is not implemented, so we will skip it for now

@pytest.fixture(autouse=True)
def _clear_config_singleton():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()

@pytest.fixture(scope="module")
def config():
    return Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="recovered_config.yml"
    )

@pytest.mark.asyncio
async def test_azure_openai_connection(config):
    model = AzureOpenAIModel(config)
    prompt = "Say hello."
    try:
        response = await model.generate(prompt=prompt)
        assert isinstance(response, str) and response, "No response from Azure OpenAI"
    except Exception as e:
        pytest.fail(f"Azure OpenAI connection failed: {e}")

@pytest.mark.asyncio
async def test_azure_search_connection(config):
    fetcher = AzureSearchFetcher(config)
    try:
        # Use a minimal valid query and filter (adjust as needed for your data)
        result = await fetcher.fetch(query="test", filter="*")
        assert "results" in result, "No results key in Azure Search response"
    except Exception as e:
        pytest.fail(f"Azure Search connection failed: {e}")

# @pytest.mark.asyncio
# async def test_sql_server_connection(config):
#     # You may need to adjust these attribute names based on your config
#     try:
#         server = config.app.other.sql_server
#         database = config.app.other.sql_database
#         fetcher = SQLServerFetcher(server=server, database=database)
#         result = await fetcher.fetch(sql="SELECT 1 AS one;")
#         assert "rows" in result and result["rows"], "No rows returned from SQL Server"
#     except Exception as e:
#         pytest.fail(f"SQL Server connection failed: {e}")

# @pytest.mark.asyncio
# async def test_postgres_connection(config):
#     try:
#         dsn = getattr(config, "postgres_dsn", os.getenv("POSTGRES_DSN"))
#         if not dsn:
#             pytest.skip("No Postgres DSN configured")
#         fetcher = PostgresFetcher(dsn=dsn)
#         result = await fetcher.fetch(sql="SELECT 1 AS one;")
#         assert "rows" in result and result["rows"], "No rows returned from Postgres"
#     except Exception as e:
#         pytest.fail(f"Postgres connection failed: {e}")

# @pytest.mark.asyncio
# async def test_rest_api_connection(config):
#     try:
#         fetcher = RestAPIFetcher(base_url="https://jsonplaceholder.typicode.com", token="", config=config)
#         result = await fetcher.fetch(route="posts", params={"userId": 1}, processor="noop")
#         assert "data" in result, "No data returned from REST API"
#     except Exception as e:
#         pytest.fail(f"REST API connection failed: {e}")

# Blob Storage fetcher is now implemented!
from rag_shared.core.fetchers.blob_storage import BlobStorageFetcher

@pytest.mark.asyncio
async def test_blob_storage_connection(config):
    """Test Azure Blob Storage connection."""
    try:
        # Skip if no blob storage config
        if not config.app.storage or not config.app.storage.blob_storage:
            pytest.skip("No blob storage configuration found")
            
        fetcher = BlobStorageFetcher(config)
        
        # Test with a simple blob fetch (you may need to adjust blob_name)
        # Use file mapping mode instead of direct blob_name (no default container)
        result = await fetcher.fetch(
            filename="test.txt",
            encoding="utf-8"
        )
        
        assert "source" in result and result["source"] == "blob_storage"
        assert "blob_name" in result
        
        # Clean up
        await fetcher.close()
        
    except Exception as e:
        pytest.fail(f"Blob Storage connection failed: {e}")

@pytest.mark.asyncio
async def test_blob_storage_file_mapping(config):
    """Test blob storage file type mapping functionality"""
    from rag_shared.core.fetchers.blob_storage import BlobStorageFetcher
    from rag_shared.utils.config_dataclasses import FileTypeMappingConfig
    from unittest.mock import MagicMock
    
    # Create a mock config with file mappings
    mock_config = MagicMock()
    mock_config.app.storage.blob_storage.container_name = "test-container"
    mock_config.app.storage.blob_storage.connection_string = "test-connection"
    mock_config.app.storage.blob_storage.file_mappings = FileTypeMappingConfig(
        base_container="conversionfiles",
        file_type_mappings={
            ".pdf": "pdf/raw",
            ".txt": "transcripts/raw"
        },
        default_directory="misc/other"
    )
    
    fetcher = BlobStorageFetcher(mock_config)
    
    # Test PDF file mapping
    container, blob_name = fetcher.get_blob_path_for_file("document.pdf")
    assert container == "conversionfiles"
    assert blob_name == "pdf/raw/document.pdf"
    
    # Test with subdirectory
    container, blob_name = fetcher.get_blob_path_for_file("report.pdf", "2024/Q1")
    assert container == "conversionfiles"
    assert blob_name == "pdf/raw/2024/Q1/report.pdf"
    
    # Test unknown file type (uses default)
    container, blob_name = fetcher.get_blob_path_for_file("unknown.xyz")
    assert container == "conversionfiles"
    assert blob_name == "misc/other/unknown.xyz"
    
    # Test Windows path normalization
    container, blob_name = fetcher.get_blob_path_for_file("file.txt", "folder\\subfolder")
    assert container == "conversionfiles"
    assert blob_name == "transcripts/raw/folder/subfolder/file.txt"

@pytest.mark.asyncio
async def test_storage_config_validation():
    """Test that storage configuration is properly structured."""
    from rag_shared.utils.config_dataclasses import StorageConfig, BlobStorageConfig
    
    # Test blob storage config
    blob_config = BlobStorageConfig(
        account_name="teststorage",
        use_managed_identity=True
    )
    
    storage_config = StorageConfig(blob_storage=blob_config)
    
    assert storage_config.blob_storage is not None
    assert storage_config.blob_storage.account_name == "teststorage"
    assert storage_config.blob_storage.use_managed_identity is True
    assert storage_config.blob_storage.endpoint_suffix == "core.windows.net"  # default

# --- LLM module tests ---
import asyncio
from rag_shared.core.models.base import LLMModel
from rag_shared.core.prompt_builders.chat_prompt import ChatPromptBuilder
from rag_shared.core.orchestrators.rag_orchestrator import RagOrchestrator

class DummyLLM(LLMModel):
    async def generate(self, prompt=None, messages=None, **kwargs):
        # Echoes the prompt or messages for testing
        if messages:
            return "|".join(m["content"] for m in messages if "content" in m)
        return prompt or "dummy"

def test_llm_generate_prompt():
    llm = DummyLLM()
    result = asyncio.run(llm.generate(prompt="test prompt"))
    assert "test prompt" in result

def test_llm_generate_messages():
    llm = DummyLLM()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"}
    ]
    result = asyncio.run(llm.generate(messages=messages))
    assert "sys" in result and "hi" in result

def test_chat_prompt_builder():
    builder = ChatPromptBuilder()
    fetched = {"AzureSearchFetcher": {"results": [
        {"text": "A"}, {"text": "B"}, {"text": "C"}
    ]}}
    user_question = "What is this?"
    prompt = builder.build(fetched, user_question)
    assert isinstance(prompt, list)
    assert any("A" in m["content"] for m in prompt)

@pytest.mark.asyncio
async def test_rag_orchestrator_llm_integration(config):
    # Use DummyLLM to test RagOrchestrator logic
    fetcher = AzureSearchFetcher(config)
    llm = DummyLLM()
    builder = ChatPromptBuilder()
    orchestrator = RagOrchestrator(
        fetchers=[fetcher],
        model=llm,
        prompt_builder=builder,
        config=config
    )
    # Patch fetcher.fetch to return dummy data
    async def dummy_fetch(**kwargs):
        return {"results": [{"text": "foo"}]}
    fetcher.fetch = dummy_fetch
    result = await orchestrator("What is foo?", fetch_args=None, history=None)
    assert "foo" in result["answer"]
