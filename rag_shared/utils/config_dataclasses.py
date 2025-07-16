from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

# ---------------------------
# YAML-config dataclasses
# ---------------------------

# Optional Dataclasses for each section of the YAML config

@dataclass
class AzureSearchParams:
    query: Optional[str] = None
    filter: Optional[str] = None
    top_k: Optional[int] = 5
    skip: Optional[int] = 0
    include_total_count: Optional[bool] = False
    facets: Optional[List[str]] = None
    highlight_fields: Optional[List[str]] = None
    search_fields: Optional[List[str]] = None
    select_fields: Optional[List[str]] = None
    semantic: Optional[bool] = False
    semantic_config: Optional[str] = None
    vector_search: Optional[bool] = False
    hybrid: Optional[float] = None
    vector_field: Optional[str] = "contentVector"

@dataclass
class RestAPIParams:
    base_url: Optional[str] = None
    token: Optional[str] = None

@dataclass
class RestAPIFetcherConfig:
    processor: Optional[str] = None
    params: Optional[RestAPIParams] = None

@dataclass
class AzureSearchFetcherConfig:
    processor: Optional[str] = None
    params: Optional[AzureSearchParams] = None


@dataclass
class FetchersConfig:
    RestAPIFetcher: Optional[RestAPIFetcherConfig] = None
    AzureSearchFetcher: Optional[AzureSearchFetcherConfig] = None


@dataclass
class PromptConfig:
    folder:   Optional[str]       = None          # e.g. "recovered_space"
    system:   Optional[str]       = None          # e.g. "system_prompt.j2"
    defaults: List[str]           = field(default_factory=list)
    # ↑ allow multiple user/assistant templates

# ───────────────────────────────────────────────
# 2.  Generation / sampling parameters
# ───────────────────────────────────────────────
@dataclass
class GenerationParams:
    max_tokens:          Optional[int] = None
    temperature:         Optional[float] = None
    top_p:               Optional[float] = None
    frequency_penalty:   Optional[float] = None
    presence_penalty:    Optional[float] = None
    stop:                Optional[List[str]] = field(default_factory=list)  # Fixed: use factory for mutable list
    repetition_penalty:  Optional[float] = None
    n:                   Optional[int] = None
    stream:              Optional[bool] = None
    logit_bias:          Optional[Dict[str, int]] = field(default_factory=dict)  # Fixed: use factory for mutable dict


@dataclass
class LLMConfig:
    type:           str
    deployment:     str
    model_name:     str  
    endpoint:       str
    api_base_url:   str
    api_version:    str
    api_key:        str = field(default="", repr=False)

    processor:      Optional[str]           = None
    prompts:        Optional[PromptConfig]  = None
    params:         Optional[GenerationParams] = None



@dataclass
class EmbeddingModelConfig:
    url:         str
    vectorizer_base_url: str
    deployment:  str
    model_name:  str 
    api_version: str
    api_key:     str = field(default="", repr=False)      # secret
    endpoint:    Optional[str] = None


# ── level 2:  index settings ──────────────────────────────────────
@dataclass
class IndexConfig:
    # mandatory
    name:             str
    skillset_name:    str
    indexer_name:     str
    indexes_path:     str
    index_yml_path:   str
    vector_dim:       int
    vector_field:     str
    index_text_field: str

    # semantic search fields
    semantic_content_fields: Optional[List[str]] = None  # e.g., ["text", "content"]
    semantic_title_field: Optional[str] = None           # e.g., "title" or "questionoranswer"

    # optional
    embedding: Optional[EmbeddingModelConfig] = None

# ── level 1:  service-wide Azure AI Search block ──────────────────
@dataclass  
class AiSearchConfig:
    index:   IndexConfig 
    endpoint: str
    api_key: str = field(default="", repr=False)          # secret
     
@dataclass
class FormRecognizerConfig:
    endpoint:     str
    api_version:  Optional[str] = None
    model_id:     str            = "prebuilt-document"
    pages_per_call: Optional[int] = 2
    api_key:      str            = field(default="", repr=False)   # secret

@dataclass
class OtherConfig:
    debug: Optional[bool] = None
    log_level: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    environment: Optional[str] = None
    custom_setting: Optional[str] = None

@dataclass
class SecretsMapping:
    # Map of secret names to their attribute paths
    AzureSearchAPIKey:          List[str]
    AzureSearchEmbeddingAPIKey: List[str]
    OpenAIAPIKey:               List[str]
    FormRecognizerAPIKey:       List[str]

@dataclass
class KVSecrets:
    # Azure AI Search secrets
    search_endpoint:            Optional[str] = field(default=None, repr=False)
    search_key:                 Optional[str] = field(default=None, repr=False)
    search_embedding_url:       Optional[str] = field(default=None, repr=False)
    search_embedding_api_key:   Optional[str] = field(default=None, repr=False)
    search_embedding_api_ver:   Optional[str] = field(default=None, repr=False)

    # LLM Model secrets
    openai_endpoint:            Optional[str] = field(default=None, repr=False)
    openai_key:                 Optional[str] = field(default=None, repr=False)
    openai_model_name:          Optional[str] = field(default=None, repr=False)

@dataclass
class AppConfig:
    name: str = "DefaultApp"
    deployment: Optional[str] = None
    fetchers: Optional[FetchersConfig] = None
    llm: Optional[LLMConfig] = None
    ai_search: Optional[AiSearchConfig] = None
    form_recognizer: Optional[FormRecognizerConfig] = None
    secrets_mapping: Optional[SecretsMapping] = None
    other: Optional[OtherConfig] = None