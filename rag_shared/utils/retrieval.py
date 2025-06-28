from os import name
import openai
from typing import Optional, Any, Dict, List, Final
from typing import cast
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorQuery

from azure.search.documents.models import VectorizedQuery, VectorizableTextQuery, VectorQuery
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration
)
from rag_shared.utils.config import Config
from rag_shared.utils.config_dataclasses import IndexConfig
from datetime import datetime

class Retrieval:
    def __init__(self, *, config: Config) -> None:
        self.config: Config = config

        # Use injected secrets from config.app
        ai_search_cfg = config.app.ai_search
        llm_cfg = config.app.llm
        index_cfg = ai_search_cfg.index # type: ignore

        # Azure Cognitive Search credentials and endpoints
        self.cred: AzureKeyCredential = AzureKeyCredential(ai_search_cfg.api_key) # type: ignore
        self._index_cfg: Final[IndexConfig] = index_cfg

        self.index_client = SearchIndexClient(
            endpoint=ai_search_cfg.endpoint, # type: ignore
            credential=self.cred,
        )

        name = index_cfg.name
        assert name is not None, "config.app.ai_search.index.name must not be None"

        self.search_client = SearchClient(
            endpoint=ai_search_cfg.endpoint, # type: ignore
            index_name=name,
            credential=self.cred,
        )

        # Azure OpenAI embeddings client
        self.openai_embeddings_client = openai.AzureOpenAI(
            api_key=index_cfg.embedding.api_key, # type: ignore
            api_version=index_cfg.embedding.api_version,    # type: ignore
            azure_endpoint=index_cfg.embedding.url, # type: ignore
            azure_deployment=index_cfg.embedding.deployment # type: ignore
        )
        # Azure OpenAI chat completions client
        self.openai_client = openai.AzureOpenAI(
            api_key=llm_cfg.api_key, # type: ignore
            api_version=llm_cfg.api_version, # type: ignore
            azure_endpoint=llm_cfg.api_base_url, # type: ignore
            azure_deployment=llm_cfg.deployment # type: ignore
        )

    def _document_exists(self, doc_id: str) -> bool:
        """
        Return True if a document with the given key already exists in the index.
        """
        try:
            self.search_client.get_document(key=doc_id)
            return True
        except ResourceNotFoundError:
            return False

    def upload_documents(self, docs: List[dict]) -> List[dict]:
        """
        Upsert documents into the Azure Search index based on their 'id'.
        New docs are created; existing docs are updated.

        Returns the indexing result for all provided documents.
        """
        # Validate documents have 'id'
        for doc in docs:
            if 'id' not in doc:
                raise ValueError("Each document must have an 'id' field.")

        # Perform upsert (mergeOrUpload) for all docs in one call
        result = self.search_client.merge_or_upload_documents(documents=docs)
        return [r.__dict__ for r in result]

    def embed(
        self,
        docs: List[dict]
    ) -> List[dict]:
        # 1) Collect texts
        texts = [d[self._index_cfg.index_text_field] for d in docs]

        # 2) Call the Azure OpenAI embeddings API
        resp = self.openai_embeddings_client.embeddings.create(
            model=self._index_cfg.embedding.deployment, # type: ignore
            input=texts
        )
        if not resp or not hasattr(resp, 'data') or not isinstance(resp.data, list):
            raise ValueError("Invalid response from embeddings API.")
        if not resp.data:
            raise ValueError("No embeddings returned.")

        # 3) Attach embeddings back to the docs
        for doc, emb in zip(docs, resp.data):
            doc[self._index_cfg.vector_field] = emb.embedding

        return docs

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        skip: int = 0,
        filter: Optional[str] = None,
        order_by: Optional[List[str]] = None,
        facets: Optional[List[str]] = None,
        highlight_fields: Optional[List[str]] = None,
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
        include_total_count: bool = False,
        semantic: bool = False,
        semantic_config: Optional[str] = None,
        vector_search: bool = False,
        hybrid: Optional[float] = None,
        vector_field: str = "contentVector",
    ) -> Dict[str, Any]:
        is_vector_only = vector_search and hybrid is None
        is_hybrid = hybrid is not None

        # Prepare vector queries if needed
        vector_queries = []
        if is_vector_only or is_hybrid:
            # 1) Build a dummy doc for your embedder using your configured text‐field name
            dummy = { self._index_cfg.index_text_field: query }
            docs_with_vec = self.embed([ dummy ])
            
            # 2) Pull the embedding back out from your configured vector‐field name
            vec = docs_with_vec[0][ self._index_cfg.vector_field ]

            # 3) Build the VectorQuery, again using your configured vector field
            vector_query = VectorizedQuery(
                vector=vec,
                k_nearest_neighbors=top_k,
                fields=self._index_cfg.vector_field
            )
            if is_hybrid:
                vector_query.weight = hybrid

            vector_queries.append(vector_query)

        search_text = '*' if is_vector_only else query
        qtype = 'semantic' if semantic else None
        sem_conf = semantic_config if semantic else None
        effective_highlight_fields = None if is_vector_only else highlight_fields

        results = self.search_client.search(
            search_text=search_text,
            filter=filter,
            order_by=order_by,
            facets=facets,
            highlight_fields=','.join(effective_highlight_fields) if effective_highlight_fields else None,
            search_fields=search_fields,
            select=select_fields,
            top=top_k,
            skip=skip,
            include_total_count=include_total_count,
            vector_queries=vector_queries,
            query_type=qtype,
            semantic_configuration_name=sem_conf,
        )
        
        hits = []
        for doc in results:
            record = dict(doc)
            record['_score'] = doc.get('@search.score')
            hits.append(record)

        return {
            'results': hits,
            'total_count': results.get_count() if include_total_count else None
        }

if __name__ == '__main__':

    config = Config(
        key_vault_name="RecoveredSpacesKV",
        config_filename="recovered_config.yml",
        config_folder="resources/configs"
    )
    # Mock document for testing
    doc_list = [
        {
            "id": "test-0001",
            "timestamp": datetime.utcnow().timestamp(),  # float seconds
            "text": (
                "Anastasiya (00:01.242)\n"
                "Hi everyone and welcome back to our podcast. … TESTING\n"
            ),
            "questionoranswer": "question",
            "speaker": "Anastasiya",
            "video_url": None,
            "keyword": None,
            "topic": None,
            "filename": "anastasiya-intro"
        }
    ]

    # Upload mock document
    retrieval = Retrieval(config=config)
    results = retrieval.upload_documents(doc_list)
    print("\nUpload results:")
    for r in results:
        print(f"  id={r['key']}, succeeded={r['succeeded']}, status={r['status_code']}")

    # Retrieve and display stored documents
    print("\nRetrieving uploaded documents:")
    for r in results:
        if r['succeeded']:
            stored = retrieval.search_client.get_document(key=r['key'])
            print(f"\nDocument ID: {r['key']}")
            print(f"  text: {stored['text']}")
            # Print additional fields if present
            for field in ['speaker', 'questionoranswer', 'filename']:
                if field in stored:
                    print(f"  {field}: {stored.get(field)}")