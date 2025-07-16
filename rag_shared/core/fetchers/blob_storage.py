import asyncio
from typing import Any, Dict, Optional
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential
from rag_shared.core.fetchers.base import DataFetcher
from rag_shared.utils.config import Config

class BlobStorageFetcher(DataFetcher):
    """
    Fetch data from Azure Blob Storage.
    
    Supports both managed identity and account key authentication.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._client: Optional[BlobServiceClient] = None
        
        if not config.app.storage or not config.app.storage.blob_storage:
            raise ValueError("Blob storage configuration is required")
            
        self.blob_config = config.app.storage.blob_storage
        
    async def _get_client(self) -> BlobServiceClient:
        """Get or create the blob service client."""
        if self._client is None:
            if self.blob_config.use_managed_identity:
                # Use managed identity (recommended for production)
                credential = DefaultAzureCredential()
                account_url = f"https://{self.blob_config.account_name}.blob.{self.blob_config.endpoint_suffix}"
                self._client = BlobServiceClient(account_url=account_url, credential=credential)
            elif self.blob_config.connection_string:
                # Use connection string
                self._client = BlobServiceClient.from_connection_string(self.blob_config.connection_string)
            elif self.blob_config.account_key:
                # Use account key
                account_url = f"https://{self.blob_config.account_name}.blob.{self.blob_config.endpoint_suffix}"
                self._client = BlobServiceClient(account_url=account_url, credential=self.blob_config.account_key)
            else:
                raise ValueError("No valid authentication method configured for blob storage")
        
        return self._client
    
    @classmethod
    def build_args(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build arguments for fetch() from context.
        
        Expected context keys:
        - blob_name: Name of the blob to fetch
        - container_name: Optional override of container name
        """
        blob_name = context.get("blob_name")
        if not blob_name:
            return {}
            
        return {
            "blob_name": blob_name,
            "container_name": context.get("container_name"),
            "encoding": context.get("encoding", "utf-8")
        }
    
    async def fetch(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch blob content from Azure Blob Storage.
        
        Args:
            blob_name: Name of the blob to fetch
            container_name: Optional override of container name
            encoding: Text encoding (default: utf-8)
        
        Returns:
            Dict containing blob metadata and content
        """
        blob_name = kwargs.get("blob_name")
        if not blob_name:
            raise ValueError("blob_name is required")
            
        container_name = kwargs.get("container_name") or self.blob_config.container_name
        encoding = kwargs.get("encoding", "utf-8")
        
        client = await self._get_client()
        blob_client = client.get_blob_client(container=container_name, blob=blob_name)
        
        try:
            # Get blob properties
            properties = await blob_client.get_blob_properties()
            
            # Download blob content
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            
            # Try to decode as text if encoding is specified
            if encoding:
                try:
                    content = content.decode(encoding)
                except UnicodeDecodeError:
                    # Keep as bytes if can't decode
                    pass
            
            return {
                "source": "blob_storage",
                "container_name": container_name,
                "blob_name": blob_name,
                "content": content,
                "content_type": properties.content_settings.content_type,
                "size": properties.size,
                "last_modified": properties.last_modified.isoformat() if properties.last_modified else None,
                "etag": properties.etag,
                "metadata": properties.metadata or {}
            }
            
        except Exception as e:
            return {
                "source": "blob_storage",
                "container_name": container_name,
                "blob_name": blob_name,
                "error": str(e),
                "content": None
            }
    
    async def close(self):
        """Close the blob service client."""
        if self._client:
            await self._client.close()


# Example usage
if __name__ == "__main__":
    import asyncio
    from rag_shared.utils.config import Config
    
    async def main():
        cfg = Config(
            key_vault_name="your-keyvault",
            config_folder="resources/configs", 
            config_filename="storage_example.yml"
        )
        
        fetcher = BlobStorageFetcher(cfg)
        
        # Fetch a text file
        result = await fetcher.fetch(
            blob_name="documents/sample.txt",
            encoding="utf-8"
        )
        
        print(f"Fetched blob: {result['blob_name']}")
        print(f"Content type: {result['content_type']}")
        print(f"Size: {result['size']} bytes")
        print(f"Content preview: {str(result['content'])[:100]}...")
        
        await fetcher.close()
    
    asyncio.run(main())