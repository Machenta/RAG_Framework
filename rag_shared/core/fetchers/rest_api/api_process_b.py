from rag_shared.core.fetchers.registry import register_processor

@register_processor("RestAPIFetcher", "extract_active_users")
def extract_active_users(raw_json: dict) -> list[dict]:
    """
    From the same keyed dict, only keep users where attrs['active'] is True,
    still flattening out the id field.
    """
    return [
        {"id": user_id, **attrs}
        for user_id, attrs in raw_json.items()
        if attrs.get("active", False)
    ]