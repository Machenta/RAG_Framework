
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

# ---------------------------
# YAML-config dataclasses
# ---------------------------


class AzureSearchParams(BaseModel):
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


class RestAPIParams(BaseModel):
    base_url: Optional[str] = None
    token: Optional[str] = None


class RestAPIFetcherConfig(BaseModel):
    processor: Optional[str] = None
    params: Optional[RestAPIParams] = None


class AzureSearchFetcherConfig(BaseModel):
    processor: Optional[str] = None
    params: Optional[AzureSearchParams] = None



class FetchersConfig(BaseModel):
    RestAPIFetcher: Optional[RestAPIFetcherConfig] = None
    AzureSearchFetcher: Optional[AzureSearchFetcherConfig] = None



class PromptConfig(BaseModel):
    folder:   Optional[str]       = None          # e.g. "recovered_space"
    system:   Optional[str]       = None          # e.g. "system_prompt.j2"
    defaults: List[str]           = Field(default_factory=list)
    # ↑ allow multiple user/assistant templates

# ───────────────────────────────────────────────
# 2.  Generation / sampling parameters
# ───────────────────────────────────────────────

class GenerationParams(BaseModel):
    max_tokens:          Optional[int] = None
    temperature:         Optional[float] = None
    top_p:               Optional[float] = None
    frequency_penalty:   Optional[float] = None
    presence_penalty:    Optional[float] = None
    stop:                Optional[List[str]] = Field(default_factory=list)
    repetition_penalty:  Optional[float] = None
    n:                   Optional[int] = None
    stream:              Optional[bool] = None
    logit_bias:          Optional[Dict[str, int]] = Field(default_factory=dict)



class LLMConfig(BaseModel):
    type:           str
    deployment:     str
    model_name:     str  
    endpoint:       str
    api_base_url:   str
    api_version:    str
    api_key:        str = Field(default="", repr=False)

    processor:      Optional[str]           = None
    prompts:        Optional[PromptConfig]  = None
    params:         Optional[GenerationParams] = None




class EmbeddingModelConfig(BaseModel):
    url:         str
    vectorizer_base_url: str
    deployment:  str
    model_name:  str 
    api_version: str
    api_key:     str = Field(default="", repr=False)      # secret
    endpoint:    Optional[str] = None


# ── level 2:  index settings ──────────────────────────────────────

class IndexConfig(BaseModel):
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

class AiSearchConfig(BaseModel):
    index:   IndexConfig 
    endpoint: str
    api_key: str = Field(default="", repr=False)          # secret

class FormRecognizerConfig(BaseModel):
    endpoint:     str
    api_version:  Optional[str] = None
    model_id:     str            = "prebuilt-document"
    pages_per_call: Optional[int] = 2
    api_key:      str            = Field(default="", repr=False)   # secret


class FileTypeMappingConfig(BaseModel):
    """Configuration for mapping file types to storage containers and directories."""
    base_container: str = "conversionfiles"
    file_type_mappings: Dict[str, str] = Field(default_factory=lambda: {
        ".pdf": "pdf/raw",
        ".txt": "transcripts/raw", 
        ".docx": "docs/raw",
        ".xlsx": "data/raw",
        ".csv": "data/raw",
        ".json": "data/raw",
        ".pptx": "presentations/raw",
        ".mp4": "video/raw",
        ".mp3": "audio/raw",
        ".wav": "audio/raw",
        ".png": "images/raw",
        ".jpg": "images/raw",
        ".jpeg": "images/raw"
    })
    default_directory: str = "other/raw"
    case_insensitive: bool = True

    def get_container_path(self, file_extension: str) -> str:
        if self.case_insensitive:
            file_extension = file_extension.lower()
        subdirectory = self.file_type_mappings.get(file_extension, self.default_directory)
        return f"{self.base_container}/{subdirectory}"

    def get_blob_name(self, filename: str, subdirectory: str = "") -> str:
        import os
        _, ext = os.path.splitext(filename)
        directory_path = self.file_type_mappings.get(
            ext.lower() if self.case_insensitive else ext, 
            self.default_directory
        )
        if subdirectory:
            subdirectory = subdirectory.replace('\\', '/')
            directory_path = f"{directory_path}/{subdirectory}"
        return f"{directory_path}/{filename}"


class BlobStorageConfig(BaseModel):
    account_name: str
    container_name: str
    account_key: Optional[str] = Field(default="", repr=False)
    connection_string: Optional[str] = Field(default="", repr=False)
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"
    file_mappings: Optional[FileTypeMappingConfig] = Field(default_factory=FileTypeMappingConfig)


class FileShareConfig(BaseModel):
    account_name: str
    share_name: str
    directory_path: Optional[str] = None
    account_key: Optional[str] = Field(default="", repr=False)
    connection_string: Optional[str] = Field(default="", repr=False)
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"


class TableStorageConfig(BaseModel):
    account_name: str
    table_name: str
    account_key: Optional[str] = Field(default="", repr=False)
    connection_string: Optional[str] = Field(default="", repr=False)
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"


class QueueStorageConfig(BaseModel):
    account_name: str
    queue_name: str
    account_key: Optional[str] = Field(default="", repr=False)
    connection_string: Optional[str] = Field(default="", repr=False)
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"


class StorageConfig(BaseModel):
    blob_storage: Optional[BlobStorageConfig] = None
    file_share: Optional[FileShareConfig] = None
    table_storage: Optional[TableStorageConfig] = None
    queue_storage: Optional[QueueStorageConfig] = None
    default_account_name: Optional[str] = None
    default_resource_group: Optional[str] = None
    default_subscription_id: Optional[str] = None


class OtherConfig(BaseModel):
    debug: Optional[bool] = None
    log_level: Optional[str] = None
    telemetry_enabled: Optional[bool] = None
    environment: Optional[str] = None
    custom_setting: Optional[str] = None


class SecretsMapping(BaseModel):
    AzureSearchAPIKey:          List[str]
    AzureSearchEmbeddingAPIKey: List[str]
    OpenAIAPIKey:               List[str]
    FormRecognizerAPIKey:       List[str]
    BlobStorageAccountKey:      Optional[List[str]] = None
    BlobStorageConnectionString: Optional[List[str]] = None
    FileShareAccountKey:        Optional[List[str]] = None
    FileShareConnectionString:  Optional[List[str]] = None
    TableStorageAccountKey:     Optional[List[str]] = None
    TableStorageConnectionString: Optional[List[str]] = None
    QueueStorageAccountKey:     Optional[List[str]] = None
    QueueStorageConnectionString: Optional[List[str]] = None


class KVSecrets(BaseModel):
    search_endpoint:            Optional[str] = None
    search_key:                 Optional[str] = None
    search_embedding_url:       Optional[str] = None
    search_embedding_api_key:   Optional[str] = None
    search_embedding_api_ver:   Optional[str] = None
    openai_endpoint:            Optional[str] = None
    openai_key:                 Optional[str] = None
    openai_model_name:          Optional[str] = None



class ExperimentVariant(BaseModel):
    name: str
    prompt_path: str
    weight: int = 50

class Experiment(BaseModel):
    name: str
    status: str = "inactive"
    traffic_split: int = 0
    variants: Dict[str, ExperimentVariant] = Field(default_factory=dict)
    success_metrics: List[str] = Field(default_factory=list)
    enabled: bool = False

class ExperimentsConfig(BaseModel):
    enabled: bool = False  # Global flag to enable/disable all experiments
    experiments: Dict[str, Experiment] = Field(default_factory=dict)
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name with proper type hints for autocomplete."""
        return self.experiments.get(name)
    
    def is_experiment_active(self, name: str) -> bool:
        """
        Check if a specific experiment is active.
        
        An experiment is active only if:
        1. Global experiments are enabled (self.enabled = True)
        2. The specific experiment exists
        3. The specific experiment is enabled (experiment.enabled = True)
        4. The specific experiment status is "active"
        """
        if not self.enabled:
            return False
        
        experiment = self.get_experiment(name)
        if not experiment:
            return False
            
        return (
            experiment.enabled and 
            experiment.status.lower() == "active"
        )
    
    def get_active_experiments(self) -> Dict[str, Experiment]:
        """Get all currently active experiments."""
        if not self.enabled:
            return {}
        
        return {
            name: exp for name, exp in self.experiments.items()
            if exp.enabled and exp.status.lower() == "active"
        }
    
    def __getitem__(self, name: str) -> Experiment:
        """Allow dictionary-style access with proper type hints."""
        return self.experiments[name]
    
    def __contains__(self, name: str) -> bool:
        """Allow 'name in experiments' syntax."""
        return name in self.experiments


class APIContactConfig(BaseModel):
    """API contact information"""
    name: Optional[str] = None
    email: Optional[str] = None
    url: Optional[str] = None


class APILicenseConfig(BaseModel):
    """API license information"""
    name: Optional[str] = None
    url: Optional[str] = None


class APIServerConfig(BaseModel):
    """API server configuration"""
    url: str
    description: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None


class APIConfig(BaseModel):
    """API housekeeping configuration for FastAPI"""
    version: Optional[str] = "1.0.0"
    description: Optional[str] = "RAG Service API for enhanced AI responses"
    contact: Optional[APIContactConfig] = None
    license: Optional[APILicenseConfig] = None
    terms_of_service: Optional[str] = None
    servers: Optional[List[APIServerConfig]] = None
    
    # FastAPI-specific settings
    docs_url: Optional[str] = "/api/docs"
    openapi_url: Optional[str] = "/api/openapi.json"
    redoc_url: Optional[str] = "/api/redoc"
    
    # API routing settings
    prefix: Optional[str] = "/api"
    enabled_endpoints: Optional[List[str]] = Field(default_factory=lambda: ["rag", "chat", "health"])
    
    # Template and static file settings
    templates_dir: Optional[str] = "templates"
    static_dir: Optional[str] = None
    static_url: Optional[str] = None


class AppConfig(BaseModel):
    name: str = "DefaultApp"
    deployment: Optional[str] = None
    api: Optional[APIConfig] = None
    fetchers: Optional[FetchersConfig] = None
    llm: Optional[LLMConfig] = None
    ai_search: Optional[AiSearchConfig] = None
    storage: Optional[StorageConfig] = None
    form_recognizer: Optional[FormRecognizerConfig] = None
    secrets_mapping: Optional[SecretsMapping] = None
    experiments: Optional[ExperimentsConfig] = None
    other: Optional[OtherConfig] = None