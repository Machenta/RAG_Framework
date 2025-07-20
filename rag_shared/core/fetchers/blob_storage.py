import asyncio
import os
import logging
from typing import Any, Dict, Optional, List
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential
from rag_shared.core.fetchers.base import DataFetcher
from rag_shared.utils.config import Config

logger = logging.getLogger(__name__)

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
            else:
                raise ValueError("Only managed identity authentication is supported for blob storage")
        
        return self._client
    
    @classmethod
    def build_args(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build arguments for fetch() from context.
        
        Expected context keys:
        - blob_name: Name of the blob to fetch, OR
        - filename: Filename to auto-map to correct container path
        - container_name: Optional override of container name
        - subdirectory: Optional additional subdirectory within the file type directory
        """
        blob_name = context.get("blob_name")
        filename = context.get("filename")
        
        if not blob_name and not filename:
            return {}
        
        return {
            "blob_name": blob_name,
            "filename": filename,
            "container_name": context.get("container_name"),
            "subdirectory": context.get("subdirectory", ""),
            "encoding": context.get("encoding", "utf-8")
        }
    
    def get_blob_path_for_file(self, filename: str, subdirectory: str = "") -> tuple[str, str]:
        """
        Get the container and blob path for a file using file type mappings.
        
        Args:
            filename: Original filename with extension
            subdirectory: Optional additional subdirectory
            
        Returns:
            Tuple of (container_name, blob_name)
        """
        if not self.blob_config.file_mappings:
            # Fallback to configured container and simple path
            return self.blob_config.container_name, filename
            
        # Get the full container path (includes base container + file type directory)
        container_path = self.blob_config.file_mappings.get_container_path(
            os.path.splitext(filename)[1]
        )
        
        # Extract just the base container name (first part)
        container_name = container_path.split('/')[0]
        
        # Get the blob name (directory structure + filename)
        blob_name = self.blob_config.file_mappings.get_blob_name(filename, subdirectory)
        
        return container_name, blob_name
    
    def get_prompt_blob_path(self, prompt_type: str, filename: str) -> tuple[str, str]:
        """
        Get the container and blob path for a prompt file.
        
        Args:
            prompt_type: Type of prompt ('system_prompts', 'response_templates', 'experiments')
            filename: Prompt filename
            
        Returns:
            Tuple of (container_name, blob_name)
        """
        if not self.blob_config.prompts_storage:
            raise ValueError("Prompts storage configuration is not available")
            
        container_name = self.blob_config.prompts_storage.container_name
        directories = self.blob_config.prompts_storage.directories or {}
        
        directory = directories.get(prompt_type, prompt_type)
        blob_name = f"{directory}/{filename}"
        
        return container_name, blob_name
    
    async def fetch(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch blob content from Azure Blob Storage.
        
        Args:
            blob_name: Direct blob name to fetch, OR
            filename: Filename to auto-map using file type mappings, OR
            prompt_type: Type of prompt ('system_prompts', 'response_templates', 'experiments') with filename
            container_name: Optional override of container name
            subdirectory: Optional additional subdirectory (when using filename)
            encoding: Text encoding (default: utf-8)
        
        Returns:
            Dict containing blob metadata and content
        """
        blob_name = kwargs.get("blob_name")
        filename = kwargs.get("filename")
        prompt_type = kwargs.get("prompt_type")
        subdirectory = kwargs.get("subdirectory", "")
        encoding = kwargs.get("encoding", "utf-8")
        
        # Determine container and blob name
        if blob_name:
            # Direct blob name provided
            container_name = kwargs.get("container_name") or self.blob_config.container_name
            final_blob_name = blob_name
        elif prompt_type and filename:
            # Prompt file - use prompts storage configuration
            container_name, final_blob_name = self.get_prompt_blob_path(prompt_type, filename)
        elif filename:
            # Use file mapping to determine paths
            container_name, final_blob_name = self.get_blob_path_for_file(filename, subdirectory)
        else:
            raise ValueError("Either blob_name, or filename, or (prompt_type + filename) must be provided")
        
        client = await self._get_client()
        blob_client = client.get_blob_client(container=container_name, blob=final_blob_name)
        
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
                "blob_name": final_blob_name,
                "original_filename": filename,  # Track original filename if used
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
                "blob_name": final_blob_name,
                "original_filename": filename,
                "error": str(e),
                "content": None
            }
    
    async def list_prompts(self, prompt_type: str) -> List[str]:
        """
        List all prompt files of a specific type.
        
        Args:
            prompt_type: Type of prompt ('system_prompts', 'response_templates', 'experiments')
            
        Returns:
            List of prompt filenames in the specified type directory
        """
        if not self.blob_config.prompts_storage:
            return []
        
        # Get container and base path for prompt type
        prompts_config = self.blob_config.prompts_storage
        container_name = prompts_config.container_name or self.blob_config.container_name
        
        directories = prompts_config.directories or {}
        if prompt_type not in directories:
            return []
        
        prefix = directories[prompt_type]
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        
        try:
            # Get container client for the prompts container
            client = await self._get_client()
            prompts_container_client = client.get_container_client(container_name)
            
            blob_list = []
            async for blob in prompts_container_client.list_blobs(name_starts_with=prefix):
                # Extract filename from full path
                blob_name = blob.name
                if blob_name.startswith(prefix):
                    filename = blob_name[len(prefix):]
                    # Only return files, not subdirectories
                    if filename and '/' not in filename:
                        blob_list.append(filename)
            
            return sorted(blob_list)
        except Exception as e:
            logger.error(f"Error listing prompts of type {prompt_type}: {e}")
            return []
    
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