import os
import yaml
import logging
import copy
from typing import Optional, Dict, Any, Type, TypeVar, cast
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

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
        config_folder:   sub‑package under ``rag_shared.resources`` (e.g. "configs")
        config_filename: YAML file inside that folder (e.g. "app_config.yml")
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
        self._config_path = os.path.join(self.config_folder, self.config_filename)

        # --- load YAML config from filesystem ---
        if not os.path.exists(self._config_path):
            raise FileNotFoundError(
                f"Configuration file not found: '{self._config_path}'"
            )
        with open(self._config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}

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
            "has_storage": self.app.storage is not None,
            "secrets_configured": self.secrets_mapping is not None,
        }
        
        # Add experiment status (safe information)
        if self.app.experiments:
            summary["experiments"] = {}
            for exp_name, experiment in self.app.experiments.experiments.items():
                summary["experiments"][exp_name] = {
                    "enabled": experiment.enabled,
                    "status": experiment.status,
                    "traffic_split": experiment.traffic_split
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
    import pprint

    # 1) Instantiate your Config
    cfg = Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="handbook_config.yml"
    )

    # 2) Pull everything into a dict
    full_cfg: Dict[str, Any] = cfg.to_dict()


    # 3) Pretty-print in Python
    print("=== Full Configuration ===")
    pprint.pprint(full_cfg)
    print()

    # # 4) Also show as JSON if you like
    # print("=== Configuration as JSON ===")
    # print(json.dumps(full_cfg, indent=2))
    # # ————————————————————————————————————————————————
    # # Verify singleton behavior
    # # ————————————————————————————————————————————————
    config2 = Config(key_vault_name="RecoveredSpacesKV", config_filename="handbook_config.yml", config_folder="resources/configs")
    assert cfg is config2, "Config should be a singleton!"
    print("Singleton verified: config is the same instance on second call.")

    # Print only the form recognizer config
    if cfg.app.form_recognizer:
        print("=== Form Recognizer Config ===")
        pprint.pprint(cfg.app.form_recognizer.model_dump())
    else:
        print("No Form Recognizer config found.")

    # Demonstrate different ways to access experiments with autocomplete
    print("\n=== Experiment Access Examples ===")
    
    # Method 1: Using the new get_experiment method (RECOMMENDED - has autocomplete!)
    experiment = cfg.get_experiment("prompt_optimization")
    if experiment:
        print(f"Experiment enabled status: {experiment.enabled}")
        experiment.enabled = True  # This will have autocomplete!
        print(f"After setting: {experiment.enabled}")
    
    # Method 2: Using the experiments config directly (also has autocomplete)
    if cfg.app.experiments:
        exp = cfg.app.experiments["prompt_optimization"]  # This now has autocomplete too!
        print(f"Direct access - enabled: {exp.enabled}")
    
    # Method 3: Safe access with get_experiment method
    if cfg.app.experiments and "prompt_optimization" in cfg.app.experiments:
        exp = cfg.app.experiments.get_experiment("prompt_optimization")
        if exp:
            print(f"Safe access - enabled: {exp.enabled}") 

