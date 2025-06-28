from typing import Union, List, Dict, Any
from rag_shared.core.fetchers.registry import register_processor


@register_processor("RestAPIFetcher", "flatten_user_dict")
def flatten_user_dict(raw_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    If raw_json is a dict keyed by user-id, turns it into a list of records:
      { "user123": {...}, "user456": {...} }
        → [ {"id":"user123", ...}, {"id":"user456", ...} ]

    If raw_json is already a list of dicts, just returns it unchanged
    (optionally filtering out any non-dicts).
    """
    if isinstance(raw_json, dict):
        # flatten dict-of-dicts
        return [
            {"id": user_id, **attrs}
            for user_id, attrs in raw_json.items()
        ]
    elif isinstance(raw_json, list):
        # assume each entry is already a record; filter to dicts only
        return [item for item in raw_json if isinstance(item, dict)]
    else:
        raise ValueError(f"Unsupported type for flatten_user_dict: {type(raw_json)}")
