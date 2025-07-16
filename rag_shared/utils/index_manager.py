from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast
import os

import yaml
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields, SemanticField,
    SearchIndex,
)

from rag_shared.utils.config import Config
from rag_shared.utils.config_dataclasses import IndexConfig

class IndexManager:
    """
    High-level helper around Azure Cognitive Search indexes.

    Parameters
    ----------
    config: Config
        Parsed configuration (already validated).
    base_dir: str | Path | None
        When *relative* paths are given in `config`, they are resolved against
        this directory.  Defaults to the current working directory.
    """
    def __init__(self, config: Config, *, base_dir: Optional[Union[str, Path]] = None):
        self.config = config
        self._base_dir = Path(base_dir or Path.cwd())

        # ------ 1.  Load index schema & transcripts ----------------------
        index_cfg: Optional[IndexConfig] = config.app.ai_search.index #type: ignore
        if index_cfg is None:
            raise ValueError("Config.app.index must not be None")
        if index_cfg.indexes_path is None or index_cfg.index_yml_path is None:
            raise ValueError("Both index_cfg.indexes_path and index_cfg.index_yml_path must not be None")
        
        # Validate path formats and warn about potential issues
        self._validate_config_paths(index_cfg)
        
        self.schema = self._load_index_file()

        # ------ 2.  Azure clients ---------------------------------------
        cred      = AzureKeyCredential(config.app.ai_search.api_key)  # type: ignore
        endpoint  = config.app.ai_search.endpoint  # type: ignore
        self.client         = SearchIndexClient(endpoint=endpoint, credential=cred)
        self.indexer_client = SearchIndexerClient(endpoint=endpoint, credential=cred)

        # ------ 3.  Convenience props -----------------------------------
        self.index_name    = index_cfg.name
        self.skillset_name = index_cfg.skillset_name
        self.indexer_name  = index_cfg.indexer_name

    # ------------------------------------------------------------------ #
    # Internal loader utilities                                          #
    # ------------------------------------------------------------------ #

    def _resolve_path(self, *parts: str) -> str:
        """
        Join *parts* with `self.base_dir` **unless** the first part is absolute.
        Handles both Windows and Unix-style path separators.

        Returns
        -------
        str  – Normalised, absolute path.
        """
        # Normalize path separators to avoid Windows backslash issues
        normalized_parts = []
        for part in parts:
            # Replace backslashes with forward slashes for consistent handling
            normalized_part = part.replace('\\', '/')
            normalized_parts.append(normalized_part)
        
        candidate = os.path.join(*normalized_parts)
        if os.path.isabs(candidate):
            return os.path.normpath(candidate)          # absolute already
        return os.path.normpath(os.path.join(self._base_dir, candidate))

    def _validate_config_paths(self, index_cfg: IndexConfig) -> None:
        """
        Validate and warn about path format issues in the configuration.
        
        Parameters
        ----------
        index_cfg : IndexConfig
            The index configuration to validate.
        """
        if index_cfg.indexes_path and '\\' in index_cfg.indexes_path:
            print(f"[IndexManager] Warning: Found backslashes in indexes_path '{index_cfg.indexes_path}'. "
                  f"Consider using forward slashes for cross-platform compatibility.")
        
        if index_cfg.index_yml_path and '\\' in index_cfg.index_yml_path:
            print(f"[IndexManager] Warning: Found backslashes in index_yml_path '{index_cfg.index_yml_path}'. "
                  f"Consider using forward slashes for cross-platform compatibility.")

    def _load_index_file(self) -> Dict[str, Any]:
        """
        Load *one* index definition file (YAML or JSON) whose location is declared
        in Config.app.index.{indexes_path,index_yml_path}.

        Returns
        -------
        dict – The parsed content.
        """
        idx_cfg = self.config.app.ai_search.index  # type: ignore
        if not idx_cfg or not idx_cfg.indexes_path or not idx_cfg.index_yml_path:
            raise ValueError(
                "Config.app.index, indexes_path, and index_yml_path must not be None"
            )

        full_path = self._resolve_path(str(idx_cfg.indexes_path),
                                    str(idx_cfg.index_yml_path))

        with open(full_path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        ext = os.path.splitext(full_path)[1].lower()
        print(f"[IndexManager] Loaded index file from {full_path}")

        if ext in (".yml", ".yaml"):
            return yaml.safe_load(raw) or {}
        if ext == ".json":
            return json.loads(raw)

        raise ValueError(f"Unsupported index file extension: {ext}")
    

    def exists(self, index_name: Optional[str] = None) -> bool:
        """Check if the index exists."""
        name = index_name if index_name is not None else self.index_name
        if name is None:
            raise ValueError("Index name must not be None")
        try:
            self.client.get_index(name)
            return True
        except Exception:
            return False

    def create_index(self):
        """Create or update search index using schema from passed YAML or Config."""
        if not self.index_name:
            raise ValueError("Index name must be defined in Config.app.index.name")

        fields_cfg = self.schema.get("fields", [])
        fields = []
        for f in fields_cfg:
            type_str = f.get("type")
            if type_str == "Collection":
                item_type = getattr(SearchFieldDataType, f["item_type"])
                dtype = SearchFieldDataType.Collection(item_type)
            else:
                dtype = getattr(SearchFieldDataType, type_str)

            kwargs = {
                "key": f.get("key", False),
                "searchable": f.get("searchable", False),
                "filterable": f.get("filterable", False),
                "retrievable": f.get("retrievable", True),
                "sortable": f.get("sortable", False)
            }
            if "vector" in f:
                v = f["vector"]
                kwargs.update({
                    "vector_search_dimensions": v.get("dimensions"),
                    "vector_search_profile_name": v.get("profile_name")
                })

            fields.append(SearchField(name=f.get("name"), type=dtype, **kwargs))

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="hnsw-config")
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-config",
                    vectorizer_name="azure-oai"
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="azure-oai",
                    kind="azureOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.config.app.ai_search.index.embedding.vectorizer_base_url,  # type: ignore
                        deployment_name=self.config.app.ai_search.index.embedding.deployment,  # type: ignore
                        model_name=self.config.app.ai_search.index.embedding.model_name,  # type: ignore
                        api_key=self.config.app.ai_search.index.embedding.api_key  # type: ignore
                    )
                )
            ]
        )

        semantic_cfg = SemanticConfiguration(
            name="default-semantic",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[
                    SemanticField(field_name=field_name) 
                    for field_name in (self.config.app.ai_search.index.semantic_content_fields or ["text"])  # type: ignore
                ],
                title_field=SemanticField(
                    field_name=self.config.app.ai_search.index.semantic_title_field or "questionoranswer"  # type: ignore
                ) if self.config.app.ai_search.index.semantic_title_field else None  # type: ignore
            )
        )
        semantic_search = SemanticSearch(configurations=[semantic_cfg])

        definition = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        self.client.create_or_update_index(definition)
        print(f"✅ Created/updated index '{self.index_name}'")

    def create_skillset(self):
        """
        Create or update the skillset based on config.indexer_schema.
        """
        # Implementation placeholder, uses self.config.indexer_schema
        pass


if __name__ == '__main__':
    from rag_shared.utils.config import Config
    cfg = Config(key_vault_name="RecoveredSpacesKV",
                 config_filename="handbook_config.yml",
                 config_folder="resources/configs")
    
    # check the paths are correct for the index manager
    print(f"Config loaded: {cfg}")

    #print the AI search config
    print(f"AI Search Config: {cfg.app.ai_search.endpoint}, {cfg.app.ai_search.api_key}, {cfg.app.ai_search.index.name}") #type: ignore
    idx_manager = IndexManager(config = cfg)

    print('Index name: ', idx_manager.index_name)
    print('The default index exists: ', idx_manager.exists())

    if not idx_manager.exists():
        idx_manager.create_index()
    else:
        print(f'Index {cfg.app.ai_search.index.name} already exists') #type: ignore

    # idx_manager.create_embedding_skillset()

    # val = idx_manager.indexer_client.get_skillset(cfg.app.ai_search.skillset_name) is not None

    # print(val)