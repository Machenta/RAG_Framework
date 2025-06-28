import os
import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict,Any, Type
from typing import TypeVar, Type, cast
from dacite import from_dict
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from rag_shared.utils.config_dataclasses import (
    AppConfig
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
    # If the first attribute is not 'app', assume the root is obj.app
    if hasattr(obj, "app") and attr_path and attr_path[0] != "app":
        obj = obj.app
    # If the first attribute is 'app', skip it
    if attr_path and attr_path[0] == "app":
        attr_path = attr_path[1:]
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

@dataclass
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

        # --- load YAML config from filesystem ---
        config_path = os.path.join(self.config_folder, self.config_filename)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: '{config_path}'"
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        
        self.secrets_mapping = raw.get("secrets_mapping", {})

        # Map into the AppConfig dataclass
        self.app: AppConfig = from_dict(AppConfig, raw.get("app", {}))

        # Get the screts mapping from the app config
        if not hasattr(self.app, "secrets_mapping"):
            raise ValueError("AppConfig must have a 'secrets_mapping' attribute")

        print("Secrets Mapping", self.app.secrets_mapping)

        self.secrets_mapping = self.app.secrets_mapping

        # Initialize Python logging level from config.other.log_level
        lvl = (
            self.app.other.log_level.upper()
            if (self.app.other and self.app.other.log_level)
            else "INFO"
        )
        logging.basicConfig(level=getattr(logging, lvl, logging.INFO))

        # KV client will be lazily created on first use
        self._kv_client: Optional[SecretClient] = None

        # map to dataclass structure
        self.app: AppConfig = from_dict(AppConfig, raw.get("app", {}))

        # ── 3) init logging level from config ────────────────────────────
        log_level = (
            (self.app.other.log_level.upper() if self.app.other and self.app.other.log_level else "INFO")
        )
        logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

        # Key Vault client is lazy‑created when first secret is requested
        self._kv_client: Optional[SecretClient] = None

        # ── 4) inject secrets from Key Vault into AppConfig ─────────────
        self._inject_secrets()


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
        # If secrets_mapping is a dataclass or namespace, convert to dict
        mapping = (
            vars(self.secrets_mapping)
            if hasattr(self.secrets_mapping, "__dict__")
            else self.secrets_mapping
        )
        if not isinstance(mapping, dict):
            logging.warning("secrets_mapping is not a dict, skipping secret injection.")
            return
        for secret_name, attr_path in mapping.items():
            # Support attribute paths as lists or tuples
            if not isinstance(attr_path, (list, tuple)):
                continue
            secret_value = self._get_secret(secret_name)
            if secret_value is not None:
                set_nested_attr(self, list(attr_path), secret_value)

    # --- Key Vault backed properties ---
    # @cached_property
    # def kv(self) -> KVSecrets:
    #     # map of secret property names to vault keys
    #     mapping = {
    #         'endpoint': 'AzureSearchEndpoint',
    #         'api_key': 'AzureSearchAPIKey',
    #         'search_embedding_url': 'AzureSearchEmbeddingURL',
    #         'search_embedding_api_key': 'AzureSearchEmbeddingAPIKey',
    #         'endpoint': 'OpenAIEndpoint',
    #         'api_key': 'OpenAIModelAPIKey',
    #     }
    #     secrets = {}
    #     for prop, secret_name in mapping.items():
    #         try:
    #             secrets[prop] = self._get_secret(secret_name)
    #         except Exception:
    #             secrets[prop] = ""
    #     return KVSecrets(**secrets)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {'app': asdict(self.app)}
        # Define keys you want to hide
        sensitive_keys = {"api_key", "key", "endpoint"}
        data = mask_sensitive_keys(data, sensitive_keys)
        return data

    def __repr__(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)




if __name__ == "__main__":
    import pprint

    # 1) Instantiate your Config
    cfg = Config(
        key_vault_name="RecoveredSpacesKV",
        config_folder="resources/configs",
        config_filename="recovered_config.yml"
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
    config2 = Config(key_vault_name="RecoveredSpacesKV", config_filename="recovered_config.yml", config_folder="resources/configs")
    assert cfg is config2, "Config should be a singleton!"
    print("Singleton verified: config is the same instance on second call.")