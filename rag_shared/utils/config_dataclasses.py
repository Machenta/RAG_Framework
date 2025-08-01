
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
    # Fields to extract as metadata from search results
    metadata_fields: Optional[List[str]] = Field(
        default_factory=lambda: ["filename", "video_url", "timestamp", "speaker", "topic", "keyword", "questionoranswer"],
        description="List of fields to extract as metadata from Azure Search results"
    )



class FetchersConfig(BaseModel):
    RestAPIFetcher: Optional[RestAPIFetcherConfig] = None
    AzureSearchFetcher: Optional[AzureSearchFetcherConfig] = None



class PromptBlobConfig(BaseModel):
    """Configuration for blob storage-based prompts"""
    # List of system prompt filenames; directory comes from storage config
    system_prompts: List[str] = Field(
        default_factory=list,
        description="List of system prompt filenames"
    )
    response_templates: List[str] = Field(
        default_factory=list,
        description="List of response template filenames"
    )
    # Optional versioning support
    version: Optional[str] = None

class PromptFilesystemConfig(BaseModel):
    """Configuration for filesystem-based prompts (legacy)"""
    folder: Optional[str] = None          # e.g. "recovered_space"
    system: Optional[str] = None          # e.g. "system_prompt.j2"
    defaults: List[str] = Field(default_factory=list)

class PromptConfig(BaseModel):
    source: str = "filesystem"  # "filesystem" or "blob_storage"
    blob_config: Optional[PromptBlobConfig] = None
    filesystem_config: Optional[PromptFilesystemConfig] = None
    
    # Validation to ensure proper config based on source
    def model_post_init(self, __context):
        if self.source == "blob_storage" and not self.blob_config:
            raise ValueError("blob_config is required when source is 'blob_storage'")
        elif self.source == "filesystem" and not self.filesystem_config:
            raise ValueError("filesystem_config is required when source is 'filesystem'")

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
    use_managed_identity: bool = Field(default=True, description="Use System-Assigned Managed Identity for authentication")

    processor:      Optional[str]           = None
    prompts:        Optional[PromptConfig]  = None
    params:         Optional[GenerationParams] = None




class EmbeddingModelConfig(BaseModel):
    url:         str
    vectorizer_base_url: str
    deployment:  str
    model_name:  str 
    api_version: str
    api_key:     str = Field(default="", repr=False)      # secret - fallback only
    endpoint:    Optional[str] = None
    use_managed_identity: bool = Field(default=True, description="Use System-Assigned Managed Identity for authentication")


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
    api_key: str = Field(default="", repr=False)          # secret - fallback only
    use_managed_identity: bool = Field(default=True, description="Use System-Assigned Managed Identity for authentication")
    # Fields to extract as metadata from search results
    metadata_fields: Optional[List[str]] = Field(
        default_factory=lambda: ["filename", "video_url", "timestamp", "speaker", "topic", "keyword", "questionoranswer"],
        description="List of fields to extract as metadata from search results"
    )

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


class PromptsStorageConfig(BaseModel):
    """Prompts storage configuration within blob storage"""
    container_name: str = "prompts"
    directories: Optional[Dict[str, str]] = Field(default_factory=lambda: {
        "system_prompts": "system_prompts",
        "response_templates": "response_templates",
        "experiments": "experiments"
    })

class BlobStorageConfig(BaseModel):
    account_name: str
    use_managed_identity: bool = True  # Always use managed identity
    endpoint_suffix: Optional[str] = "core.windows.net"
    prompts_storage: Optional[PromptsStorageConfig] = None  # Nested prompts config
    file_mappings: Optional[FileTypeMappingConfig] = Field(default_factory=FileTypeMappingConfig)


class FileShareConfig(BaseModel):
    account_name: str
    share_name: str
    directory_path: Optional[str] = None
    use_managed_identity: bool = True  # Always use managed identity
    endpoint_suffix: Optional[str] = "core.windows.net"


class TableStorageConfig(BaseModel):
    account_name: str
    table_name: str
    use_managed_identity: bool = True  # Always use managed identity
    endpoint_suffix: Optional[str] = "core.windows.net"


class QueueStorageConfig(BaseModel):
    account_name: str
    queue_name: str
    use_managed_identity: bool = True  # Always use managed identity
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
    # Note: Storage services use system-assigned managed identity, no secrets needed


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
    # Basic API metadata
    title: Optional[str] = "RAG Service"
    version: Optional[str] = "1.0.0"
    description: Optional[str] = "RAG Service API for enhanced AI responses"
    contact: Optional[APIContactConfig] = None
    license: Optional[APILicenseConfig] = None
    terms_of_service: Optional[str] = None
    servers: Optional[List[APIServerConfig]] = None
    
    # FastAPI-specific documentation settings
    docs_url: Optional[str] = "/api/docs"
    openapi_url: Optional[str] = "/api/openapi.json"
    redoc_url: Optional[str] = "/api/redoc"
    
    # API routing settings
    prefix: Optional[str] = "/api"
    enabled_endpoints: Optional[List[str]] = Field(default_factory=lambda: ["rag", "chat", "health"])
    
    # Template and static file settings
    templates_dir: Optional[str] = Field(default="templates", description="Directory for Jinja2 templates")
    static_dir: Optional[str] = Field(default=None, description="Directory for static files")
    static_url: Optional[str] = Field(default=None, description="URL prefix for static files")
    
    # Additional FastAPI configuration
    debug: Optional[bool] = None  # Can override app-level debug setting
    host: Optional[str] = "0.0.0.0"
    port: Optional[int] = 8000
    reload: Optional[bool] = None  # Auto-reload for development
    workers: Optional[int] = 1  # Number of worker processes
    
    # Security settings
    cors_enabled: Optional[bool] = True
    cors_origins: Optional[List[str]] = Field(default_factory=lambda: ["*"])
    cors_methods: Optional[List[str]] = Field(default_factory=lambda: ["GET", "POST"])
    cors_headers: Optional[List[str]] = Field(default_factory=lambda: ["*"])
    
    # Request/response settings
    max_request_size: Optional[int] = 16 * 1024 * 1024  # 16MB default
    timeout: Optional[int] = 30  # Request timeout in seconds
    
    # Logging and monitoring
    access_log: Optional[bool] = True
    log_level: Optional[str] = None  # Can override app-level log setting

    # Features block (for feature toggles)
    features: Optional["FeaturesConfig"] = None


class FeaturesConfig(BaseModel):
    feedback_collection_enabled: bool = False
    ab_testing_enabled: bool = False
    custom_metrics_enabled: bool = False

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