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
class FileTypeMappingConfig:
    """Configuration for mapping file types to storage containers and directories."""
    
    # Base container name (all files go here)
    base_container: str = "conversionfiles"
    
    # File extension to subdirectory mapping (will be joined with base_container)
    # Using forward slashes - they'll be normalized for the target platform
    file_type_mappings: Dict[str, str] = field(default_factory=lambda: {
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
    
    # Default directory for unknown file types
    default_directory: str = "other/raw"
    
    # Whether to use file extension case-insensitive matching
    case_insensitive: bool = True
    
    def get_container_path(self, file_extension: str) -> str:
        """
        Get the full container path for a given file extension.
        
        Args:
            file_extension: File extension (e.g., ".pdf", ".txt")
            
        Returns:
            Full path like "conversionfiles/pdf/raw"
        """
        if self.case_insensitive:
            file_extension = file_extension.lower()
            
        # Get the subdirectory, or use default
        subdirectory = self.file_type_mappings.get(file_extension, self.default_directory)
        
        # Join with base container using forward slashes (cross-platform safe)
        return f"{self.base_container}/{subdirectory}"
    
    def get_blob_name(self, filename: str, subdirectory: str = "") -> str:
        """
        Generate a blob name for a file, including directory structure.
        
        Args:
            filename: Original filename
            subdirectory: Optional additional subdirectory
            
        Returns:
            Blob name like "pdf/raw/document.pdf" or "pdf/raw/2024/document.pdf"
        """
        import os
        
        # Extract file extension
        _, ext = os.path.splitext(filename)
        
        # Get the base directory path
        directory_path = self.file_type_mappings.get(
            ext.lower() if self.case_insensitive else ext, 
            self.default_directory
        )
        
        # Add subdirectory if provided
        if subdirectory:
            # Normalize subdirectory path separators
            subdirectory = subdirectory.replace('\\', '/')
            directory_path = f"{directory_path}/{subdirectory}"
        
        # Return the full blob name (always use forward slashes for blob storage)
        return f"{directory_path}/{filename}"

@dataclass
class BlobStorageConfig:
    account_name: str
    container_name: str
    account_key: Optional[str] = field(default="", repr=False)  # secret, can use managed identity instead
    connection_string: Optional[str] = field(default="", repr=False)  # secret alternative
    use_managed_identity: Optional[bool] = True  # prefer managed identity over keys
    endpoint_suffix: Optional[str] = "core.windows.net"  # for sovereign clouds
    
    # File type mapping configuration
    file_mappings: Optional[FileTypeMappingConfig] = field(default_factory=FileTypeMappingConfig)

@dataclass
class FileShareConfig:
    account_name: str
    share_name: str
    directory_path: Optional[str] = None  # subdirectory within the share
    account_key: Optional[str] = field(default="", repr=False)  # secret
    connection_string: Optional[str] = field(default="", repr=False)  # secret alternative
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"

@dataclass
class TableStorageConfig:
    account_name: str
    table_name: str
    account_key: Optional[str] = field(default="", repr=False)  # secret
    connection_string: Optional[str] = field(default="", repr=False)  # secret alternative
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"

@dataclass
class QueueStorageConfig:
    account_name: str
    queue_name: str
    account_key: Optional[str] = field(default="", repr=False)  # secret
    connection_string: Optional[str] = field(default="", repr=False)  # secret alternative
    use_managed_identity: Optional[bool] = True
    endpoint_suffix: Optional[str] = "core.windows.net"

@dataclass
class StorageConfig:
    # Different storage services
    blob_storage: Optional[BlobStorageConfig] = None
    file_share: Optional[FileShareConfig] = None
    table_storage: Optional[TableStorageConfig] = None
    queue_storage: Optional[QueueStorageConfig] = None
    
    # Global storage account settings (if using a single storage account)
    default_account_name: Optional[str] = None
    default_resource_group: Optional[str] = None
    default_subscription_id: Optional[str] = None

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
    
    # Storage-related secrets (optional)
    BlobStorageAccountKey:      Optional[List[str]] = None
    BlobStorageConnectionString: Optional[List[str]] = None
    FileShareAccountKey:        Optional[List[str]] = None
    FileShareConnectionString:  Optional[List[str]] = None
    TableStorageAccountKey:     Optional[List[str]] = None
    TableStorageConnectionString: Optional[List[str]] = None
    QueueStorageAccountKey:     Optional[List[str]] = None
    QueueStorageConnectionString: Optional[List[str]] = None

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
    storage: Optional[StorageConfig] = None
    form_recognizer: Optional[FormRecognizerConfig] = None
    secrets_mapping: Optional[SecretsMapping] = None
    other: Optional[OtherConfig] = None