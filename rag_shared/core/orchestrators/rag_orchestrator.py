import asyncio
import logging as log
from typing import Sequence, Dict, Any, List, Optional

from rag_shared.core.fetchers.base import DataFetcher
from rag_shared.core.models.base import LLMModel
from rag_shared.core.prompt_builders.base import PromptBuilder
from rag_shared.utils.config import Config
from rag_shared.core.fetchers.registry import get_processor

class RagOrchestrator:
    def __init__(
        self,
        fetchers: Sequence[DataFetcher],
        model: LLMModel,
        prompt_builder: PromptBuilder,
        config: Config,
        default_proc: str = "default",
        system_prompt: Optional[str] = None
    ):
        self.fetchers       = fetchers
        self.model          = model
        self.prompt_builder = prompt_builder
        self.default_proc   = default_proc
        self.config         = config
        self.system_prompt  = system_prompt

    async def __call__(
        self,
        user_question: str,
        fetch_args: Dict[str, Dict[str, Any]] | None = None,
        history:      List[Dict[str, str]]        | None = None,
        **model_kwargs: Any
    ) -> Dict[str, Any]:

        fetch_args = fetch_args or {}
        history    = history or []

        # ── 1. collect raw data from every fetcher ──────────────────
        gathered: Dict[str, Dict[str, Any]] = {}

        if self.fetchers:
            print("Step 1: Fetching data from all sources...")

            async def _one(fetcher):
                name = fetcher.__class__.__name__
                args = fetch_args.get(name, {})

                try:
                    raw = await fetcher.fetch(**args)
                except ValueError as exc:               # ← fetcher signalled “no data”
                    log.warning("%s returned no data: %s", name, exc)
                    return name, {}                     # empty dict keeps pipeline alive
                except Exception as exc:                # ← network, auth, etc.
                    log.exception("%s failed: %s", name, exc)
                    return name, {}

                # ── post-process if we got something ────────────────
                if not raw:
                    log.info("%s produced an empty payload.", name)
                    return name, {}

                # choose processor: arg-override → YAML → default
                proc_name = (
                    args.get("processor")
                    or getattr(self.config.app.fetchers.AzureSearchFetcher, "processor", None) #type: ignore
                    or self.default_proc
                )
                if not proc_name:
                    log.warning("No processor configured for %s; skipping.", name)
                    return name, {}

                processed = get_processor(name, proc_name)(raw)
                return name, processed

            # schedule them concurrently
            pairs = await asyncio.gather(*[_one(f) for f in self.fetchers])
            gathered = {k: v for k, v in pairs if v}    # drop empties
        # else: gathered stays {}

        # 2 ─ craft prompt or chat messages
        print("Step 2: Building prompt from fetched data")
        print(f"[RagOrchestrator] Gathered data: {gathered}")
        built = self.prompt_builder.build(gathered, user_question)
        print(f"[RagOrchestrator] Generated prompt/messages:\n{built}\n")

        # 3 ─ LLM call with memory
        print("Step 3: Calling LLM with prompt and history")
        messages: List[Dict[str, str]] = list(history)

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if isinstance(built, list):
            messages.extend(built)
        else:
            messages.append({"role": "user", "content": built})
        print(f"[RagOrchestrator] Full chat messages:\n{messages}\n")

        response = await self.model.generate(messages=messages, **model_kwargs)
        print(f"[RagOrchestrator] Model response:\n{response}\n")

        # 4 - Extract metadata dynamically from AzureSearchFetcher using its own config
        metadata: List[Dict[str, Any]] = []
        azure = gathered.get("AzureSearchFetcher", {}) or {}
        
        # Get metadata fields from fetcher config
        fields: List[str] = []
        if (
            self.config.app
            and self.config.app.fetchers
            and self.config.app.fetchers.AzureSearchFetcher
            and self.config.app.fetchers.AzureSearchFetcher.metadata_fields
        ):
            fields = self.config.app.fetchers.AzureSearchFetcher.metadata_fields  # type: ignore
        
        print(f"[RagOrchestrator] Extracting metadata for fields: {fields}")
        
        for doc in azure.get("results", []) or []:
            meta: Dict[str, Any] = {}
            
            # Extract configured metadata fields with fallbacks
            for field in fields:
                if field in doc:
                    meta[field] = doc.get(field)
                else:
                    # Provide sensible fallbacks for missing fields
                    if field == "filename":
                        meta[field] = doc.get("source", doc.get("title", "Unknown Document"))
                    elif field == "video_url":
                        meta[field] = doc.get("url", doc.get("source_url", None))
                    elif field == "timestamp":
                        meta[field] = doc.get("time", doc.get("created_time", None))
                    elif field == "chunk_index":
                        meta[field] = doc.get("chunk", doc.get("block_id", 0))
                    elif field == "speaker":
                        meta[field] = doc.get("author", doc.get("presenter", "Unknown Speaker"))
                    elif field == "topic":
                        meta[field] = doc.get("category", doc.get("subject", "General"))
                    elif field == "keyword":
                        meta[field] = doc.get("keywords", doc.get("tags", None))
                    elif field == "questionoranswer":
                        meta[field] = doc.get("qa_type", doc.get("content_type", "content"))
                    else:
                        meta[field] = None
            
            # Always include search score if available
            if "_score" in doc:
                meta["_score"] = doc["_score"]
            
            # Format source citation for easy reference
            source = meta.get("filename", "Unknown Document")
            speaker = meta.get("speaker", "Unknown Speaker")
            topic = meta.get("topic", "General Topic")
            
            # Create a meaningful citation based on available metadata
            if meta.get("video_url"):
                timestamp_info = f" at {meta.get('timestamp', 'unknown time')}" if meta.get('timestamp') else ""
                meta["citation"] = f"{source} - {speaker} ({topic}){timestamp_info}"
            else:
                chunk_info = f" (Chunk {meta.get('chunk_index', 'unknown')})" if meta.get('chunk_index') is not None else ""
                meta["citation"] = f"{source} - {speaker}{chunk_info}"
            
            # Add content preview for debugging
            text_preview = doc.get("text", "")[:100] + "..." if len(doc.get("text", "")) > 100 else doc.get("text", "")
            meta["content_preview"] = text_preview
            
            metadata.append(meta)

        print(f"[RagOrchestrator] Extracted {len(metadata)} metadata entries")
        for i, meta in enumerate(metadata):
            print(f"[RagOrchestrator] Metadata {i+1}: {meta.get('citation', 'No citation')}")

        # 5 - Build updated history
        new_history = []
        # preserve system prompt in returned history
        if self.system_prompt:
            new_history.append({"role": "system", "content": self.system_prompt})
        new_history.extend(history)
        new_history.append({"role": "user",      "content": built if isinstance(built, str) else ""})
        new_history.append({"role": "assistant", "content": response})

        # 6 - Return answer, metadata, and updated history
        return {
            "answer":   response,
            "metadata": metadata,
            "history":  new_history
        }



if __name__ == "__main__":
    import asyncio
    import os
    from rag_shared.utils.config import Config
    from rag_shared.core.fetchers.azure_search.azure_search import AzureSearchFetcher
    from rag_shared.core.models.azure_openai import AzureOpenAIModel
    from rag_shared.core.prompt_builders.template import TemplatePromptBuilder

    KEY_VAULT_NAME = os.getenv("KEY_VAULT_NAME", "RecoveredSpacesKV")
    CONFIG_PATH    = os.getenv("CONFIG_PATH",    "configs")
    CONFIG_FILE    = os.getenv("CONFIG_FILE",    "recovered_config.yml")

    # 1 - Load your config
    cfg = Config(key_vault_name="RecoveredSpacesKV", config_filename="recovered_config.yml", config_folder="resources/configs")

    # 2 - Instantiate the Azure Search fetcher
    azure_fetcher = AzureSearchFetcher(config=cfg)

    with open(os.path.join("resources", "prompts", "system_prompt.j2"),encoding="utf-8") as fh:
        system_prompt = fh.read()

    # 3 - Instantiate Model
    llm_model = AzureOpenAIModel(
        cfg,
        system_prompt=system_prompt,
        default_max_tokens=200,
        default_temperature=0.2
    )

    # 4 - Load the default prompt template and create a PromptBuilder
    default_tpl = open(os.path.join("resources/prompts", "default_prompt_with_json.j2")).read()
    builder     = TemplatePromptBuilder(default_tpl)
    # 5 - Build the orchestrator with your fetcher, model, and prompt builder
    orchestrator = RagOrchestrator(
        fetchers       = [azure_fetcher],
        model          = llm_model,
        prompt_builder = builder,
        config= cfg,
    )

    # 6 - Define the user question and the fetch_args for AzureSearchFetcher
    user_question = "Who is David?"
    fetch_args = {
        "AzureSearchFetcher": {
            "query": user_question,
            "filter": "",
            "top_k": 5,
            "include_total_count": True,
            "facets": ["speaker,count:5", "topic"],
            "highlight_fields": ["text"],
            "select_fields": [
                "id", "filename", "video_url", "timestamp", "chunk_index", "speaker", 
                "text", "topic", "keyword", "questionoranswer", "block_id", "part", "tokens"
            ],
            "vector_search": True
        }
    }

    # 7 - Kick off the orchestration
    async def main():
        print("\n===== ORCHESTRATION DEMO (Real Services) =====\n")
        print(f"User question: {user_question}")
        print(f"Fetch arguments:\n{fetch_args}\n")

        result = await orchestrator(user_question, fetch_args)

        print("===== ORCHESTRATION COMPLETE =====")
        print(f"Final result:\n{result}\n")

    asyncio.run(main())