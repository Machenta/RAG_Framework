# RAG_Framework (Rag Framework)

## Overview

- **RAG_Framework** is a modular, production-ready Retrieval-Augmented Generation (RAG) framework designed for enterprise use.  
- Integrates with Azure services (OpenAI, Cognitive Search, Key Vault, SQL, Blob Storage, etc.), supports multi-source data fetching, and provides robust configuration and secret management.  
- Python-based, agnostic, configurable, and secure—simplifies building AI-powered applications by combining user input with contextual data from multiple sources and orchestrating LLM calls with rich prompt templates.  
- Built for extensibility, testability, and secure deployment across multiple environments, supporting single-turn and multi-turn chat, context injection, and prompt engineering.

---

## Table of Contents

1. [Features](#features)  
2. [Key Components](#key-components)  
3. [Architecture Overview](#architecture-overview)  
4. [Tech Stack](#tech-stack)  
5. [Directory Structure](#directory-structure)  
6. [Prerequisites](#prerequisites)  
7. [Setup & Installation](#setup--installation)  
8. [Configuration](#configuration)  
9. [Usage Examples](#usage-examples)  
10. [API Reference](#api-reference)  
11. [Running Tests](#running-tests)  
12. [Extending the Framework](#extending-the-framework)  
13. [Azure Best Practices](#azure-best-practices)  
14. [Deployment](#deployment)  
15. [Troubleshooting](#troubleshooting)  
16. [Contributing](#contributing)  
17. [License](#license)  

---

## Features

- **RAG Orchestration**: Single-turn and multi-turn chat with persistent history, context injection, and prompt engineering.  
- **Agnostic LLM Abstraction**: Plug in Azure OpenAI, OpenAI, or custom LLM providers via a uniform interface.  
- **Azure Native Integrations**: Azure OpenAI, Cognitive Search, Key Vault, SQL, Blob Storage, and more.  
- **Modular Fetchers**: Async fetchers for REST APIs, Azure Search, Postgres, SQL Server, file systems, etc.  
- **Configurable Prompt Builders**: Chat-style, composite, and Jinja2-based template builders.  
- **Centralized YAML Configuration**: Single file with environment-specific overrides and secret mappings.  
- **Secure Secret Management**: Lazy-loading from Azure Key Vault or environment variables; managed identity support.  
- **Singleton Config**: Ensures consistent configuration across the app.  
- **FastAPI Endpoints**: Production-ready routes for `/v1/rag` and `/v1/chat`.  
- **Async/Await Support**: High-performance, non-blocking operations throughout.  

---

## Key Components

### 1. Configuration & Secrets
- `rag_shared/utils/config.py`: Loads YAML config, injects secrets from Azure Key Vault, supports singleton pattern.
- `rag_shared/utils/config_dataclasses.py`: Dataclasses for all config sections (fetchers, LLM, search, etc.).

### 2. Data Fetchers
- Azure Cognitive Search: `azure_search/azure_search.py` or `azure_ai_search_fetcher.py`  
- REST API: `rest_api/rest_api.py` (async `httpx`)  
- Postgres & SQL Server: `postgres.py`, `sql_server.py` (managed identity)  
- Blob Storage / File System: `blob_storage.py` (stub)  
- Processors: Registered post-fetch transformations (flattening, filtering).

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

### 6. API (FastAPI)
- `api/routes.py`: Exposes `/v1/rag` and `/v1/chat`, handles request/response.

### 7. Utilities
- Logging, timing, helpers in `utils/`.

### 8. Testing
- `tests/test_azure_connections.py`: Azure SDK connection tests.  
- `tests/test_config.py`: YAML loading, secret injection, singleton.  
- Additional tests for fetchers, prompt builders, orchestrators.

---

## Architecture Overview

```text
User 
 └─> API (/v1/rag, /v1/chat)
       └─> Orchestrator
             ├─> Fetchers (Search, REST, SQL, Postgres, Blob)
             │      ↳ External services → raw data
             ├─> Processors → transform data
             ├─> PromptBuilder → prompt/messages
             └─> LLMModel → AzureOpenAI/OpenAI → response
       └─> Response (answer, context, metadata, history)
```

On startup, `ConfigLoader` reads `resources/configs/*.yml`, resolves secrets from Key Vault or env vars, and provides a typed config. The orchestrator then concurrently fetches data, applies processors, builds prompts, calls the LLM, and returns results with metadata.

---

## Tech Stack

- Python 3.10+  
- `asyncio`, `httpx` for async I/O  
- PyYAML for config  
- Pydantic or dataclasses for schemas  
- Azure SDKs (`azure-identity`, `azure-keyvault-secrets`, `azure-search-documents`)  
- Jinja2 for prompt templates  
- FastAPI + Uvicorn for web service  
- pytest + pytest-asyncio for testing  

---

## Directory Structure

```
└── 📁RAG_Framework
    └── 📁rag_shared
        └── 📁core
            └── 📁deployment
                ├── rest_api_processors.py
            └── 📁fetchers
                └── 📁azure_search
                    ├── azure_search.py
                    ├── noop.py
                └── 📁rest_api
                    ├── api_process_a.py
                    ├── api_process_b.py
                    ├── rest_api.py
                ├── __init__.py
                ├── base.py
                ├── blob_storage.py
                ├── postgres.py
                ├── registry.py
                ├── sql_server.py
            └── 📁models
                ├── azure_openai.py
                ├── base.py
            └── 📁orchestrators
                ├── chat_orchestrator.py
                ├── rag_orchestrator.py
            └── 📁prompt_builders
                ├── base.py
                ├── chat_prompt.py
                ├── composite.py
                ├── json_prompt.py
                ├── template.py
        └── 📁utils
            ├── __init__.py
            ├── config_dataclasses.py
            ├── config.py
            ├── index_manager.py
            ├── retrieval.py
        ├── __init__.py
    └── 📁resources
        └── 📁AI_search_indexes
            ├── transcripts.yml
        └── 📁configs
            ├── handbook_config.yml
            ├── recovered_config.yml
        └── 📁prompts
            └── 📁recovered_space
                ├── chat_prompt.j2
                ├── default_prompt.j2
                ├── system_prompt.j2
            ├── default_prompt_with_json.j2
            ├── default_prompt.j2
            ├── system_prompt.j2
        ├── __init__.py
    └── 📁tests
        ├── test_azure_connections.py
        ├── test_config.py
    ├── .gitignore
    ├── pyproject.toml
    ├── README.md
    └── requirements.txt
```

---

## Prerequisites

- Python 3.10+  
- `pip` or Poetry  
- Azure subscription with:  
   • Cognitive Search service  
   • Azure OpenAI resource  
   • Key Vault instance  
   • (Optional) Azure SQL  
- Local Azure credentials (`az login` or service principal)  
- Environment variables or YAML entries:  
   - `RACK_CONFIG_FILE` or `CONFIG_PATH` (default: `resources/configs/config.yaml`)  
   - `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`  
   - `KEY_VAULT_URL`  

---

## Setup & Installation

1. Clone the repo  
   ```bash
   git clone <your-repo-url>
   cd RAG_Framework
   ```

2. Create & activate a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows PowerShell
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Key Vault and YAML  
   - Place YAML files in `resources/configs/` (e.g., `config.yaml`, `dev_config.yaml`).  
   - Map secret names in YAML to Key Vault entries or env vars.

---

## Configuration

All settings live in a YAML file (`resources/configs/config.yaml` by default). Override with `RAG_CONFIG_FILE`.

### Sample `config.yaml`

```yaml
azure_openai:
  endpoint: "https://<your-openai-endpoint>.openai.azure.com/"
  api_key: "<YOUR_OPENAI_API_KEY>"
  deployment_name: "gpt-4"
  embedding_deployment_name: "text-embedding-ada-002"
  embedding_model: "text-embedding-ada-002"

azure_ai_search:
  endpoint: "https://<your-search-service>.search.windows.net"
  api_key: "<YOUR_SEARCH_API_KEY>"
  index_name: "my-index"
  api_version: "2023-07-01-Preview"
  semantic_configuration_name: "default"
  semantic_search_config:
    search_fields: ["content", "metadata"]
    semantic_fields: ["content"]

llm_config:
  provider: "azure"                           
  model: "gpt-4-32k"
  endpoint: "https://<your-openai-endpoint>.openai.azure.com/"
  api_key: "<YOUR_OPENAI_API_KEY>"
  api_version: "2023-07-01-preview"
  base_url: "/openai/deployments/"

prompts:
  fetch:
    template: "resources/prompts/fetch.j2"
  summarize:
    template: "resources/prompts/summarize.j2"

field_mappings:
  document_id: "id"
  content: "content"
  metadata:
    author: "author"
    date: "date"
    tags: ["tag1", "tag2"]

keyvault_secrets:
  - "SearchServiceApiKey"
  - "OpenAIApiKey"
```

---

## Usage Examples

### Load Configuration

```python
from rag_shared.utils.config import Config

cfg = Config(
  key_vault_name="RecoveredSpacesKV",
  config_folder="resources/configs",
  config_filename="config.yaml"
)
print(cfg.azure_ai_search.endpoint)
```

### Query Azure Cognitive Search

```python
from rack_framework.connectors.azure_ai_search_fetcher import AzureAISearchFetcher

fetcher = AzureAISearchFetcher(cfg)
results = await fetcher.fetch(query="machine learning", top_k=5)
print(results)
```

### Build Prompt & Call LLM

```python
from rag_shared.core.prompt_builders.chat_prompt import ChatPromptBuilder
from rag_shared.core.models.azure_openai import AzureOpenAIModel

builder = ChatPromptBuilder()
model = AzureOpenAIModel(cfg)
prompt_msgs = builder.build(user_input="What is RAG?", context_snippets=[])
response = await model.generate(prompt_msgs)
print(response)
```

### Single-Turn Orchestration

```python
import asyncio
from rag_shared.core.orchestrators.rag_orchestrator import RagOrchestrator
from rag_shared.core.fetchers.azure_search.azure_search import AzureSearchFetcher

async def main():
    orchestrator = RagOrchestrator(
        fetchers=[AzureSearchFetcher(cfg)],
        model=AzureOpenAIModel(cfg),
        prompt_builder=ChatPromptBuilder(),
        config=cfg
    )
    result = await orchestrator.get_response("Define retrieval-augmented generation")
    print(result)
asyncio.run(main())
```

### Multi-Turn Chat

```python
import asyncio
from rag_shared.core.orchestrators.chat_orchestrator import ChatOrchestrator

async def chat_session():
    chat = ChatOrchestrator(
        fetchers=[AzureSearchFetcher(cfg)],
        model=AzureOpenAIModel(cfg),
        prompt_builder=ChatPromptBuilder(),
        config=cfg,
        system_prompt="You are a helpful assistant."
    )
    print(await chat.send_message("Hello, who are you?"))
    print(await chat.send_message("Explain RAG in simple terms."))
asyncio.run(chat_session())
```

---

## API Reference

All endpoints via FastAPI in `src/rack_framework/api/routes.py`.

### POST /v1/rag

Perform single-turn RAG.

Request:
```json
{
  "query": "Explain quantum computing",
  "top_k": 5
}
```

Response:
```json
{
  "query": "Explain quantum computing",
  "context": ["...snippets..."],
  "answer": "..."
}
```

### POST /v1/chat

Multi-turn chat.

Request:
```json
{
  "session_id": "abc123",
  "message": "What is RAG?"
}
```

Response:
```json
{
  "session_id": "abc123",
  "response": "Retrieval-augmented generation (RAG) is …"
}
```

---

## Running Tests

1. Install test deps  
   ```bash
   pip install pytest pytest-asyncio
   ```
2. Run all tests  
   ```bash
   pytest
   ```
3. Specific tests  
   ```bash
   pytest tests/test_azure_connections.py
   ```

---

## Extending the Framework

- **Fetchers**: Subclass `DataFetcher`, implement `fetch()`, register processors.  
- **Prompt Builders**: Subclass `PromptBuilder`.  
- **LLM Models**: Subclass `LLMModel`.  
- **Config**: Extend dataclasses/Pydantic definitions and update YAML.

---

## Azure Best Practices

- Manage all secrets in Azure Key Vault.  
- Use managed identity for SQL and other Azure resources.  
- Adopt async/await for scalability.  
- Validate connections with tests before deployment.

---

## Deployment

### Azure Web App

1. Push code to a GitHub repo.  
2. Create an Azure Web App for Python.  
3. Configure App Settings:  
   - `RAG_CONFIG_FILE`  
   - `KEY_VAULT_URL`  
   - `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`  
4. Enable Managed Identity or assign a Service Principal to access Key Vault.  
5. Deploy via GitHub Actions or Azure DevOps.

---

## Troubleshooting

- **FileNotFoundError: config.yaml**  
  • Verify `RAG_CONFIG_FILE` or default path.  
- **YAML parse errors**  
  • Validate with `yamllint`.  
- **Azure credential errors**  
  • Run `az login` or set `AZURE_*` env vars.  
  • Check Key Vault access policies.  
- **HTTP 401/403 from Azure**  
  • Confirm endpoint URLs and secret names.  
- **Template rendering errors**  
  • Ensure Jinja2 variables match your prompt context.

---
