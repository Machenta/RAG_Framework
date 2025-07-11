# RAG_Framework

## Overview

RAG_Framework is a modular, production-ready Retrieval-Augmented Generation (RAG) framework designed for enterprise use. It integrates with Azure services (OpenAI, Azure Search, Key Vault, SQL, Blob Storage, etc.), supports multi-source data fetching, and provides robust configuration and secret management. The framework is built for extensibility, testability, and secure deployment across multiple environments.

---

## Features

- **Retrieval-Augmented Generation (RAG) Orchestration**: Multi-turn chat, context injection, and prompt engineering.
- **Azure Native Integrations**: Azure OpenAI, Azure Search, Azure Key Vault, Azure SQL, Azure Blob Storage, and more.
- **Pluggable Data Fetchers**: Easily add new data sources (REST API, Postgres, SQL Server, etc.).
- **Configurable Prompt Builders**: Template, chat, and composite prompt builders for flexible LLM input.
- **Secure Secret Management**: Pulls secrets from Azure Key Vault and environment variables.
- **Singleton Config**: Ensures consistent configuration and secret injection across the app.
- **Extensive Test Suite**: Connection tests for all Azure services and LLM modules.
- **Async/Await Support**: All fetchers and orchestrators are async for high performance.

---

## Directory Structure

```
rag_shared/
    core/
        deployment/
        fetchers/
            azure_search/
            rest_api/
        models/
        orchestrators/
        prompt_builders/
    utils/
resources/
    AI_search_indexes/
    configs/
    prompts/
tests/
    test_azure_connections.py
    test_config.py
pyproject.toml
requirements.txt
README.md
```

---

## Key Components

### 1. Configuration & Secrets
- `rag_shared/utils/config.py`: Loads YAML config, injects secrets from Azure Key Vault, supports singleton pattern.
- `rag_shared/utils/config_dataclasses.py`: Dataclasses for all config sections (fetchers, LLM, search, etc.).

### 2. Data Fetchers
- `azure_search/azure_search.py`: Async fetcher for Azure Cognitive Search.
- `rest_api/rest_api.py`: Async fetcher for REST APIs.
- `postgres.py`, `sql_server.py`: Async fetchers for Postgres and Azure SQL (with managed identity support).
- `blob_storage.py`: (Stub) for Azure Blob Storage integration.

### 3. LLM Models
- `models/azure_openai.py`: Azure OpenAI integration, supports chat and completion APIs.
- `models/base.py`: Abstract LLMModel interface.

### 4. Orchestrators
- `orchestrators/rag_orchestrator.py`: Core RAG orchestration logic (fetch, prompt, LLM, metadata, history).
- `orchestrators/chat_orchestrator.py`: Multi-turn chat with persistent history and system prompt.

### 5. Prompt Builders
- `prompt_builders/base.py`: Abstract interface for prompt builders.
- `prompt_builders/chat_prompt.py`: Chat-style prompt builder.
- `prompt_builders/composite.py`: Composite builder for multi-source prompts.

### 6. Testing
- `tests/test_azure_connections.py`: Tests all Azure service connections and LLM module logic.
- `tests/test_config.py`: Tests config loading, secret injection, and singleton behavior.

---

<!-- Detailed Module Descriptions and Interaction Flow -->
## Module Descriptions & Interaction Flow

### Configuration & Secrets Module
- **File:** `rag_shared/utils/config.py`, `config_dataclasses.py`
- **Responsibility:** Load YAML configuration and environment variables, inject secrets from Azure Key Vault using managed identity or environment fallbacks, and provide a singleton `Config` instance.
- **Flow:** On startup, `Config` reads `resources/configs/*.yml`, maps into `AppConfig`, initializes logging, and lazily resolves secrets upon first access.

### Data Fetchers Module
- **Files:**
  - `azure_search/azure_search.py`
  - `rest_api/rest_api.py`
  - `sql_server.py`, `postgres.py`
  - `blob_storage.py` (stub)
- **Responsibility:** Implement `DataFetcher.fetch(**kwargs)` to asynchronously retrieve data from external sources.
- **Processors:** Registered via `register_processor()`, allowing post-fetch transformations (e.g. flattening, filtering).
- **Flow:** `RagOrchestrator` calls each fetcher concurrently, gathers raw data, applies the configured processor, and stores results under their fetcher key.

### Prompt Builders Module
- **Files:**
  - `prompt_builders/base.py`
  - `chat_prompt.py`, `composite.py`, `template.py`
- **Responsibility:** Transform fetched data into LLM-ready input: either a single prompt string or a list of chat message dicts (`role`, `content`).
- **Types:**
  1. **ChatPromptBuilder:** Builds a chat-style message list, injecting a system prompt and user content snippets.
  2. **CompositePromptBuilder:** Merges multiple builders, concatenating strings or merging message lists.
  3. **TemplatePromptBuilder:** (Not shown) Uses Jinja2 templates to render dynamic prompts.

### LLM Models Module
- **Files:**
  - `models/base.py` (abstract interface)
  - `azure_openai.py`
- **Responsibility:** Wrap Azure OpenAI SDK for completion and chat APIs, merge default parameters with overrides, and provide uniform `generate()` interface.
- **Flow:** Receives prompt/messages, asynchronously sends request to Azure OpenAI, and returns the trimmed response text.

### Orchestrators Module
- **Files:**
  - `orchestrators/rag_orchestrator.py`
  - `chat_orchestrator.py`
- **Responsibility:** Coordinate end-to-end RAG pipeline:
  1. **Fetch Phase:** Concurrently call each `DataFetcher`.
  2. **Process Phase:** Apply processors to raw data.
  3. **Build Phase:** Use a `PromptBuilder` to generate LLM input.
  4. **LLM Phase:** Call `LLMModel.generate()`.
  5. **Metadata & History:** Extract source metadata (e.g. document URLs, timestamps), update chat history.
- **ChatOrchestrator Extension:** Manages multi-turn chat by preserving and prepending history and system prompt on each call.

### Component Interaction Diagram
```text
User Question → RagOrchestrator → [fetchers]
                 ↳ AzureSearchFetcher → Azure Search Service → raw results
                 ↳ SQLServerFetcher → Azure SQL → rows
                 ↳ RestAPIFetcher → HTTP API → JSON
                 ↳ PostgresFetcher → Postgres DB → rows
              fetched data → processors → aggregated dict
              dict + user_question → PromptBuilder → prompt/messages
              prompt/messages → AzureOpenAIModel → LLM Response
              Response + metadata + history → returned to caller
```

---

## Setup & Installation

1. **Clone the repository**
   ```pwsh
   git clone <your-repo-url>
   cd RAG_Framework
   ```

2. **Install dependencies**
   ```pwsh
   pip install -r requirements.txt
   ```

3. **Configure Azure Key Vault and YAML config**
   - Place your config YAMLs in `resources/configs/` (see example configs).
   - Set up your Azure Key Vault and ensure your secrets match the mapping in your config.

4. **Set environment variables as needed**
   - For local dev, you can use a `.env` file or set variables in your shell.

---

## Running Tests

1. **Install test dependencies**
   ```pwsh
   pip install pytest pytest-asyncio
   ```

2. **Run all tests**
   ```pwsh
   pytest
   ```

3. **Run a specific test file**
   ```pwsh
   pytest tests/test_azure_connections.py
   ```

---

## Usage Example

See orchestrator `__main__` blocks for end-to-end usage. Example (simplified):

```python
from rag_shared.utils.config import Config
from rag_shared.core.fetchers.azure_search.azure_search import AzureSearchFetcher
from rag_shared.core.models.azure_openai import AzureOpenAIModel
from rag_shared.core.orchestrators.chat_orchestrator import ChatOrchestrator
from rag_shared.core.prompt_builders.chat_prompt import ChatPromptBuilder

cfg = Config(key_vault_name="RecoveredSpacesKV", config_folder="resources/configs", config_filename="recovered_config.yml")
fetcher = AzureSearchFetcher(cfg)
llm = AzureOpenAIModel(cfg)
builder = ChatPromptBuilder()
orchestrator = ChatOrchestrator(
    fetchers=[fetcher],
    model=llm,
    prompt_builder=builder,
    config=cfg,
    system_prompt="You are a helpful assistant."
)
result = orchestrator("What is RAG?", fetch_args=None, history=None)
print(result["answer"])
```

---

## Extending the Framework

- Add new fetchers by subclassing `DataFetcher` and registering processors.
- Add new prompt builders by subclassing `PromptBuilder`.
- Add new LLM integrations by subclassing `LLMModel`.
- Update config dataclasses and YAMLs for new settings.

---

## Azure Best Practices

- All secrets are managed via Azure Key Vault.
- Use managed identity for SQL and other Azure resources where possible.
- Async/await is used throughout for scalability.
- Tests ensure all connections are valid before deployment.

---

## License

MIT License. See `LICENSE` file for details.