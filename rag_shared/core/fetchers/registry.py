from typing import Callable, Dict
from collections import defaultdict

# { fetcher_name → { processor_name → callable } }
PROCESSOR_REGISTRY: Dict[str, Dict[str, Callable]] = defaultdict(dict)

def register_processor(fetcher: str, name: str):
    """
    Decorator to register a processor under a given fetcher type.
    E.g.:
    @register_processor('Rest_API', 'ProcessorA')
    """
    def decorator(fn: Callable):
        PROCESSOR_REGISTRY.setdefault(fetcher, {})[name] = fn
        return fn
    return decorator


def get_processor(fetcher_name: str, processor_name: str) -> Callable:
    return PROCESSOR_REGISTRY[fetcher_name][processor_name]