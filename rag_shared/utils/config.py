import os
import yaml
import logging
import copy
import asyncio
from typing import Optional, Dict, Any, Type, TypeVar, cast
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob.aio import BlobServiceClient

from rag_shared.utils.config_dataclasses import (
    AppConfig, ExperimentsConfig, LLMConfig, FetchersConfig,
    AiSearchConfig, StorageConfig, FormRecognizerConfig, OtherConfig, SecretsMapping, Experiment
)

# configure azure identity logging to warning
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

def set_nested_attr(obj, attr_path, value):
    """
    Set a nested attribute given a list of attribute names.
    If the first attribute is not 'app', automatically start from obj.app.
    Supports both list and dataclass/namespace attribute traversal.
    """
    # If path starts with 'app', navigate to it
    if attr_path and attr_path[0] == 'app':
        if hasattr(obj, 'app'):
            obj = obj.app
        attr_path = attr_path[1:]  # Skip 'app'
    
    # If path doesn't start with 'app' but obj has 'app', start from app
    elif hasattr(obj, "app") and attr_path:
        obj = obj.app
    
    for attr in attr_path[:-1]:
        # Support both dict and attribute access
        if isinstance(obj, dict):
            obj = obj.get(attr)
        else:
            obj = getattr(obj, attr, None)
        if obj is None:
            return  # Can't set deeper
    
    # Set the value
    if isinstance(obj, dict):
        obj[attr_path[-1]] = value
    else:
        setattr(obj, attr_path[-1], value)


def remove_sensitive_keys(data: Any, sensitive_keys: set) -> Any:
    """Recursively remove sensitive keys from a dict or list."""
    if isinstance(data, dict):
        return {k: remove_sensitive_keys(v, sensitive_keys)
                for k, v in data.items() if k not in sensitive_keys}
    elif isinstance(data, list):
        return [remove_sensitive_keys(i, sensitive_keys) for i in data]
    else:
        return data

def mask_sensitive_keys(data: Any, sensitive_keys: set, mask: str = "<hidden>") -> Any:
    """Recursively mask sensitive keys in a dict or list."""
    if isinstance(data, dict):
        return {k: (mask if k in sensitive_keys else mask_sensitive_keys(v, sensitive_keys, mask))
                for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_sensitive_keys(i, sensitive_keys, mask) for i in data]
    else:
        return data


T = TypeVar("T")

class SingletonMeta(type):
    _instances: Dict[Any, object] = {}

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        key = (cls, args, tuple(sorted(kwargs.items())))
        if key not in SingletonMeta._instances:
            SingletonMeta._instances[key] = super().__call__(*args, **kwargs)
        return cast(T, SingletonMeta._instances[key])

class Config(metaclass=SingletonMeta):
    """Load YAML config + Key Vault secrets.

    Args:
        key_vault_name:  Azure Key Vault name (no https://, just the name)
        config_folder:   sub‑package under ``rag_shared.resources`` (e.g. "configs") - fallback only
        config_filename: YAML file inside that folder (e.g. "app_config.yml")
    
    Environment Variables for Config Source:
        CONFIG_SOURCE: "blob_storage" or "filesystem" (default: "filesystem")
        CONFIG_STORAGE_ACCOUNT: Storage account name (required if CONFIG_SOURCE=blob_storage)
        CONFIG_CONTAINER: Container name (default: "configs")
        CONFIG_DIRECTORY: Directory within container (optional, can be empty)
        CONFIG_FILENAME: Config filename (default: from config_filename parameter)
    """
    def __init__(
        self,
        key_vault_name: str,
        *,
        config_folder: str = "configs",
        config_filename: str = "app_config.yml",
    ) -> None:
        self.key_vault_name = key_vault_name
        self.config_folder  = config_folder
        self.config_filename= config_filename

        # --- load YAML config from source (blob storage or filesystem) ---
        raw = self._load_config_sync()

        # Map into the AppConfig Pydantic model
        self.app: AppConfig = AppConfig.model_validate(raw.get("app", {}))

        # Get the secrets mapping from the app config
        if not hasattr(self.app, "secrets_mapping") or self.app.secrets_mapping is None:
            logging.warning("AppConfig missing 'secrets_mapping' attribute; skipping secret injection")
            # Don't create a default secrets mapping - let the application handle this
            # Creating empty mappings could expose security vulnerabilities
            self.secrets_mapping = None
        else:
            self.secrets_mapping = self.app.secrets_mapping

        # Initialize Python logging level from config.other.log_level
        log_level = (
            self.app.other.log_level.upper()
            if (self.app.other and self.app.other.log_level)
            else "INFO"
        )
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

        # Key Vault client is lazy‑created when first secret is requested
        self._kv_client: Optional[SecretClient] = None

        # ── inject secrets from Key Vault into AppConfig ─────────────
        self._inject_secrets()
        
        # Store the original raw config for comparison when saving
        self._original_raw = raw

    def _load_config_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for config loading that handles event loop conflicts."""
        config_source = os.getenv("CONFIG_SOURCE", "filesystem").lower()
        
        if config_source == "blob_storage":
            try:
                # Try to get existing event loop
                loop = asyncio.get_running_loop()
                # If we're in an existing event loop, we need to handle this differently
                import concurrent.futures
                import threading
                
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._load_config_from_blob())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_async)
                    return future.result()
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self._load_config_from_blob())
        else:
            return self._load_config_from_filesystem()

    async def _load_config_from_source(self) -> Dict[str, Any]:
        """Load configuration from blob storage or filesystem based on environment variables."""
        config_source = os.getenv("CONFIG_SOURCE", "filesystem").lower()
        
        if config_source == "blob_storage":
            return await self._load_config_from_blob()
        else:
            return self._load_config_from_filesystem()
    
    def _load_config_from_filesystem(self) -> Dict[str, Any]:
        """Load configuration from local filesystem (fallback method)."""
        self._config_path = os.path.join(self.config_folder, self.config_filename)
        
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(
                f"Configuration file not found: '{self._config_path}'"
            )
        
        with open(self._config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        
        logging.info(f"Loaded configuration from filesystem: {self._config_path}")
        return raw
    
    async def _load_config_from_blob(self) -> Dict[str, Any]:
        """Load configuration from Azure Blob Storage."""
        storage_account = os.getenv("CONFIG_STORAGE_ACCOUNT")
        if not storage_account:
            raise ValueError("CONFIG_STORAGE_ACCOUNT environment variable is required when CONFIG_SOURCE=blob_storage")
        
        container_name = os.getenv("CONFIG_CONTAINER", "configs")
        directory = os.getenv("CONFIG_DIRECTORY", "")
        filename = os.getenv("CONFIG_FILENAME", self.config_filename)
        
        # Construct blob path
        if directory:
            blob_name = f"{directory}/{filename}"
        else:
            blob_name = filename
        
        # Set up blob client
        credential = AsyncDefaultAzureCredential()
        account_url = f"https://{storage_account}.blob.core.windows.net"
        
        try:
            async with BlobServiceClient(account_url=account_url, credential=credential) as client:
                blob_client = client.get_blob_client(container=container_name, blob=blob_name)
                
                try:
                    # Download blob content
                    download_stream = await blob_client.download_blob()
                    content = await download_stream.readall()
                    config_text = content.decode('utf-8')
                    
                    # Parse YAML
                    raw = yaml.safe_load(config_text) or {}
                    
                    logging.info(f"Loaded configuration from blob storage: {storage_account}/{container_name}/{blob_name}")
                    
                    # Set fallback path for save operations
                    self._config_path = os.path.join(self.config_folder, filename)
                    
                    return raw
                    
                except Exception as e:
                    logging.warning(f"Failed to load config from blob storage ({storage_account}/{container_name}/{blob_name}): {e}")
                    logging.info("Falling back to filesystem configuration...")
                    return self._load_config_from_filesystem()
        finally:
            # Ensure credential is closed
            if hasattr(credential, 'close'):
                await credential.close()


    def _get_secret(self, name: str) -> str:
        if self._kv_client is None:
            vault_uri = f"https://{self.key_vault_name}.vault.azure.net"
            self._kv_client = SecretClient(vault_url=vault_uri, credential=DefaultAzureCredential())
        try:
            return self._kv_client.get_secret(name).value  # type: ignore
        except Exception:
            # fallback to ENV
            env_name = name.upper().replace('-', '_')
            val = os.getenv(env_name)
            if val:
                logging.warning(f"Falling back to env var for secret '{name}'")
                return val
            raise

    def _inject_secrets(self):
        """
        Inject secrets from Key Vault into the config object using secrets_mapping.
        Handles both dataclass and pydantic/namespace mapping objects.
        """
        if self.secrets_mapping is None:
            logging.info("No secrets mapping configured, skipping secret injection")
            return
            
        # If secrets_mapping is a Pydantic model, convert to dict
        if hasattr(self.secrets_mapping, 'model_dump'):
            mapping = self.secrets_mapping.model_dump()
        elif hasattr(self.secrets_mapping, "__dict__"):
            mapping = vars(self.secrets_mapping)
        else:
            mapping = self.secrets_mapping
            
        if not isinstance(mapping, dict):
            logging.warning("secrets_mapping is not a dict, skipping secret injection.")
            return
            
        for secret_name, attr_path in mapping.items():
            # Skip None values (optional secrets)
            if attr_path is None:
                continue
            # Support attribute paths as lists or tuples
            if not isinstance(attr_path, (list, tuple)):
                logging.warning(f"Invalid attribute path for secret '{secret_name}': {attr_path}")
                continue
            try:
                secret_value = self._get_secret(secret_name)
                if secret_value is not None:
                    set_nested_attr(self, list(attr_path), secret_value)
            except Exception as e:
                logging.error(f"Failed to inject secret '{secret_name}': {e}")
                # Continue with other secrets even if one fails

    def _merge_section(self, current_dict: Dict[str, Any], reloaded_obj: Any, section_name: str):
        """
        Generic helper to merge in-memory changes for a specific section.
        If the section exists in current (in-memory) but not in reloaded, preserve it.
        Customize or extend for deeper merging if needed (e.g., for nested dicts).
        """
        if section_name in current_dict and getattr(reloaded_obj, section_name, None) is None:
            section_data = current_dict[section_name]
            # Convert back to the appropriate Pydantic model if needed
            section_class_map = {
                'experiments': ExperimentsConfig,
                'llm': LLMConfig,
                'fetchers': FetchersConfig,
                'ai_search': AiSearchConfig,
                'storage': StorageConfig,
                'form_recognizer': FormRecognizerConfig,
                'other': OtherConfig,
                'secrets_mapping': SecretsMapping,
            }
            if section_name in section_class_map:
                merged_section = section_class_map[section_name].model_validate(section_data)
                setattr(reloaded_obj, section_name, merged_section)
            else:
                # Fallback for simple dicts or unknown sections
                setattr(reloaded_obj, section_name, section_data)

    def save(self) -> bool:
        """
        Save the current configuration back to the YAML file.
        This only saves non-secret values (secrets are managed in Key Vault).
        
        Returns:
            bool: True if the save was successful, False otherwise
        """
        try:
            # Create a backup of the original config file
            backup_path = f"{self._config_path}.bak"
            if os.path.exists(self._config_path):
                try:
                    import shutil
                    shutil.copy2(self._config_path, backup_path)
                    logging.info(f"Created backup of config file: {backup_path}")
                except Exception as e:
                    logging.warning(f"Failed to create backup of config file: {e}")
            
            # Convert the app Pydantic model to a dict with deep copy for nested structures
            app_dict = copy.deepcopy(self.app.model_dump())
            
            # Create the structure of the config file
            config_data = {"app": app_dict}
            
            # Write the updated config back to the file
            with open(self._config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)
            
            # Validate persistence by reloading the file and checking a sample key
            with open(self._config_path, 'r', encoding='utf-8') as f:
                saved = yaml.safe_load(f)
            # Example validation (customize to a key you update often)
            if 'llm' in saved.get('app', {}) and 'params' in saved['app']['llm'] and saved['app']['llm']['params'].get('max_tokens') != self.app.llm.params.max_tokens: #type: ignore
                raise ValueError("Save validation failed: Changes not persisted to file")
            
            logging.info(f"Configuration saved to {self._config_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {'app': self.app.model_dump()}
        # Define keys you want to hide
        sensitive_keys = {"api_key", "key", "endpoint"}
        data = mask_sensitive_keys(data, sensitive_keys)
        return data
    
    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name with proper type hints for autocomplete.
        
        Usage:
            experiment = cfg.get_experiment("prompt_optimization")
            if experiment:
                experiment.enabled = True  # This will have autocomplete!
        """
        if self.app.experiments:
            return self.app.experiments.get_experiment(name)
        return None

    def are_experiments_enabled(self) -> bool:
        """Check if experiments are globally enabled."""
        return (
            self.app.experiments is not None and 
            self.app.experiments.enabled
        )
    
    def is_experiment_active(self, name: str) -> bool:
        """
        Check if a specific experiment is active and ready to run.
        
        An experiment is active only if:
        1. Global experiments are enabled
        2. The specific experiment exists
        3. The specific experiment is enabled
        4. The specific experiment status is "active"
        """
        if not self.app.experiments:
            return False
        return self.app.experiments.is_experiment_active(name)
    
    def get_active_experiments(self) -> Dict[str, Experiment]:
        """Get all currently active experiments."""
        if not self.app.experiments:
            return {}
        return self.app.experiments.get_active_experiments()
    
    def enable_experiments(self, enabled: bool = True):
        """
        Enable or disable all experiments globally.
        
        Args:
            enabled: True to enable experiments, False to disable
        """
        if self.app.experiments:
            self.app.experiments.enabled = enabled

    def is_production(self) -> bool:
        """Check if running in production environment."""
        if not self.app.other or not self.app.other.environment:
            return False
        return self.app.other.environment.lower() == "production"
    
    def get_safe_config_summary(self) -> Dict[str, Any]:
        """
        Get a safe summary of configuration for monitoring/debugging.
        
        Returns only non-sensitive information suitable for logging or APIs.
        """
        summary = {
            "app_name": self.app.name,
            "deployment": self.app.deployment,
            "environment": self.app.other.environment if self.app.other else "unknown",
            "debug_mode": self.app.other.debug if self.app.other else False,
            "log_level": self.app.other.log_level if self.app.other else "INFO",
            "has_llm": self.app.llm is not None,
            "has_ai_search": self.app.ai_search is not None,
            "has_experiments": self.app.experiments is not None,
            "experiments_enabled": self.are_experiments_enabled(),
            "has_storage": self.app.storage is not None,
            "has_api_config": self.app.api is not None,
            "secrets_configured": self.secrets_mapping is not None,
        }
        
        # Add API configuration summary (safe information)
        if self.app.api:
            summary["api"] = {
                "title": self.app.api.title,
                "version": self.app.api.version,
                "description": self.app.api.description,
                "docs_url": self.app.api.docs_url,
                "openapi_url": self.app.api.openapi_url,
                "redoc_url": self.app.api.redoc_url,
                "prefix": self.app.api.prefix,
                "enabled_endpoints": self.app.api.enabled_endpoints,
                "host": self.app.api.host,
                "port": self.app.api.port,
                "cors_enabled": self.app.api.cors_enabled
                # Note: Excluding sensitive settings like cors_origins for security
            }
        
        # Add experiment status (safe information)
        if self.app.experiments:
            summary["experiments"] = {
                "global_enabled": self.app.experiments.enabled,
                "individual_experiments": {}
            }
            
            for exp_name, experiment in self.app.experiments.experiments.items():
                summary["experiments"]["individual_experiments"][exp_name] = {
                    "enabled": experiment.enabled,
                    "status": experiment.status,
                    "traffic_split": experiment.traffic_split,
                    "is_active": self.is_experiment_active(exp_name)
                    # Note: variants and success_metrics excluded for brevity
                }
        
        return summary

    def reload(self) -> bool:
        """
        Reload the configuration from disk and reinject secrets.
        Use this after making changes to the configuration file.
        
        Returns:
            bool: True if the reload was successful, False otherwise
        """
        try:
            logging.info(f"Reloading configuration from {self._config_path}")
            
            # Check if the config file exists
            if not os.path.exists(self._config_path):
                logging.error(f"Configuration file not found: '{self._config_path}'")
                return False
                
            # Load the YAML configuration
            with open(self._config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {}
            
            # Preserve current non-secret state for merging
            current_app_dict = self.app.model_dump()
            
            # Reload the app from raw (pre-merge)
            self.app = AppConfig.model_validate(raw.get("app", {}))
            
            # Generic merge for all optional sections (preserves in-memory changes if missing from disk)
            optional_sections = ['api', 'experiments', 'llm', 'fetchers', 'ai_search', 'storage', 'form_recognizer', 'other', 'secrets_mapping']
            for section in optional_sections:
                self._merge_section(current_app_dict, self.app, section)
            
            # Example: Deep merge for specific nested fields (customize as needed; this preserves max_tokens if changed in memory)
            if self.app.llm and 'llm' in current_app_dict and current_app_dict['llm'].get('params', {}).get('max_tokens') != self.app.llm.params.max_tokens:  # type: ignore
                self.app.llm.params.max_tokens = current_app_dict['llm']['params']['max_tokens']  # type: ignore
            
            # Set secrets_mapping from the merged app (ensures consistency post-merge)
            if not hasattr(self.app, "secrets_mapping") or self.app.secrets_mapping is None:
                logging.warning("AppConfig missing 'secrets_mapping' attribute; skipping secret injection")
                self.secrets_mapping = None
            else:
                self.secrets_mapping = self.app.secrets_mapping
            
            # Reinject secrets after merging and loading non-secrets
            self._inject_secrets()
            
            # Update the original raw config for future comparisons
            self._original_raw = raw
            
            # Light validation: Check if a key section loaded correctly (customize to your needs)
            if not self.app.llm:  # Example: Ensure LLM (a common section) is present
                logging.warning("Reloaded config missing 'llm' section; using in-memory fallback if available")
            
            logging.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to reload configuration: {e}")
            return False


    def __repr__(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


if __name__ == "__main__":

    if False:
        print("\n=== Filesystem Config Test ===")
        SingletonMeta._instances.clear()
        os.environ.pop("CONFIG_SOURCE", None)
        os.environ.pop("CONFIG_STORAGE_ACCOUNT", None)
        try:
            cfg_fs = Config(
                key_vault_name="RecoveredSpacesKV",
                config_folder="resources/configs",
                config_filename="handbook_config.yml"
            )
            print(f"✓ Filesystem config loaded: {cfg_fs.app.name}")
        except Exception as e:
            print(f"❌ Filesystem config failed: {e}")


    else:
        print("\n=== Blob Storage Config Test ===")
        SingletonMeta._instances.clear()
        os.environ["CONFIG_SOURCE"] = "blob_storage"
        os.environ["CONFIG_STORAGE_ACCOUNT"] = "ragrecoveredspacestorage"
        os.environ["CONFIG_CONTAINER"] = "configs"
        os.environ["CONFIG_DIRECTORY"] = "dev"
        os.environ["CONFIG_FILENAME"] = "development_config.yml"
        try:
            cfg_blob = Config(
                key_vault_name="RecoveredSpacesKV",
                config_folder="resources/configs",
                config_filename="development_config.yml"
            )
            print(f"✓ Blob storage config loaded: {cfg_blob.app.name}")

            print(f"Config: {cfg_blob.to_dict()}")
        except Exception as e:
            print(f"❌ Blob storage config failed: {e}")

        # Clean up environment variables
        for env_var in ["CONFIG_SOURCE", "CONFIG_STORAGE_ACCOUNT", "CONFIG_CONTAINER", "CONFIG_DIRECTORY", "CONFIG_FILENAME"]:
            os.environ.pop(env_var, None)

    print("\n=== Done ===")

