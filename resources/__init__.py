import os
import yaml
from importlib import resources
from typing import Dict, Any

def load_project_config(config_file: str | None = None) -> Dict[str, Any]:
    """
    Load and parse the YAML config whose filename is given by CONFIG_FILE.
    Looks for files in rag_shared/resources/configs/{CONFIG_FILE}.
    """
    # 1) pick up the config‐filename (must include “.yml”)
    cfg_file = config_file or os.getenv("CONFIG_FILE")
    if not cfg_file:
        raise RuntimeError("CONFIG_FILE must be set (or passed in)")

    # 2) load it from the packaged configs/
    with resources.open_text("rag_shared.resources.configs", cfg_file) as f:
        return yaml.safe_load(f) or {}

def load_prompt(template: str, prompt_folder: str | None = None) -> str:
    """
    Load the Jinja template named `template` from 
    rag_shared/resources/prompts/{PROMPT_FOLDER}/.
    Returns the raw template text.
    """
    # 1) pick up which folder under prompts/
    folder = prompt_folder or os.getenv("PROMPT_FOLDER")
    if not folder:
        raise RuntimeError("PROMPT_FOLDER must be set (or passed in)")

    # 2) read it from the packaged prompts/<folder>/
    pkg = f"rag_shared.resources.prompts.{folder}"
    return resources.read_text(pkg, template)

def load_index_schema(indexes_folder: str, schema_filename: str) -> Dict[str, Any]:
    """
    Load and return the index schema YAML as a dict, using os.path to locate the file.
    Prints the raw YAML before parsing.

    :param indexes_folder: path under resources/ where the schema lives (e.g. "AI_search_indexes")
    :param schema_filename: name of the YAML file (e.g. "transcripts.yml")
    """
    utils_dir = os.path.dirname(__file__)
    pkg_root = os.path.dirname(utils_dir)

    # Build the full path to the schema file
    schema_path = os.path.join(pkg_root, 'resources', indexes_folder, schema_filename)

    # Read and print the raw YAML content
    with open(schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(content)

    # Parse YAML into dict and return
    return yaml.safe_load(content) or {}