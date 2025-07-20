"""
Example of using BlobStorageFetcher with prompt management.

This example demonstrates:
1. Fetching prompts from blob storage using prompt_type and filename
2. Listing available prompts by type
3. Using the enhanced fetch method with different parameter combinations
"""

import asyncio
import os
import sys

# Add the project root to the path so we can import rag_shared
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_shared.utils.config import Config
from rag_shared.core.fetchers.blob_storage import BlobStorageFetcher


async def main():
    """
    Demonstrate blob storage fetcher with prompt management.
    """
    try:
        # Initialize configuration (you may need to adjust these values)
        config = Config(
            key_vault_name="your-key-vault-name",  # Replace with your Key Vault name
            config_folder="resources/configs",
            config_filename="handbook_config.yml"
        )
        
        # Create blob storage fetcher
        fetcher = BlobStorageFetcher(config)
        
        print("=== Blob Storage Fetcher with Prompt Management ===\n")
        
        # Example 1: List available prompts by type
        print("1. Listing available prompts by type:")
        
        prompt_types = ["system_prompts", "response_templates", "experiments"]
        for prompt_type in prompt_types:
            try:
                prompts = await fetcher.list_prompts(prompt_type)
                print(f"   {prompt_type}: {prompts}")
            except Exception as e:
                print(f"   {prompt_type}: Error - {e}")
        
        print()
        
        # Example 2: Fetch a system prompt
        print("2. Fetching a system prompt:")
        try:
            result = await fetcher.fetch(
                prompt_type="system_prompts",
                filename="default_system.j2",
                encoding="utf-8"
            )
            
            if result.get("error"):
                print(f"   Error fetching prompt: {result['error']}")
            else:
                print(f"   Successfully fetched: {result['blob_name']}")
                print(f"   Content type: {result.get('content_type', 'unknown')}")
                print(f"   Size: {result.get('size', 0)} bytes")
                if result.get("content"):
                    content_preview = str(result["content"])[:200]
                    print(f"   Content preview: {content_preview}...")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Example 3: Fetch using direct blob name
        print("3. Fetching using direct blob name:")
        try:
            result = await fetcher.fetch(
                blob_name="system_prompts/default_system.j2",
                container_name="prompts",
                encoding="utf-8"
            )
            
            if result.get("error"):
                print(f"   Error fetching blob: {result['error']}")
            else:
                print(f"   Successfully fetched: {result['blob_name']}")
                print(f"   Content type: {result.get('content_type', 'unknown')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Example 4: Fetch using filename with file type mapping
        print("4. Fetching using filename with file type mapping:")
        try:
            result = await fetcher.fetch(
                filename="sample_document.pdf",
                encoding=None  # Keep as bytes for binary files
            )
            
            if result.get("error"):
                print(f"   Error fetching file: {result['error']}")
            else:
                print(f"   Successfully fetched: {result['blob_name']}")
                print(f"   Container: {result['container_name']}")
                print(f"   Content type: {result.get('content_type', 'unknown')}")
                print(f"   Size: {result.get('size', 0)} bytes")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()
        
        # Example 5: Configuration details
        print("5. Configuration details:")
        
        if not config.app.storage or not config.app.storage.blob_storage:
            print("   Error: Blob storage not configured")
        else:
            blob_config = config.app.storage.blob_storage
            print(f"   Account name: {blob_config.account_name}")
            # No default container_name: containers defined per sub-config
            print(f"   Using managed identity: {blob_config.use_managed_identity}")
            
            if blob_config.prompts_storage:
                print(f"   Prompts container: {blob_config.prompts_storage.container_name}")
                print(f"   Prompts directories: {blob_config.prompts_storage.directories}")
            else:
                print("   Prompts storage: Not configured")
        print("\n=== Example completed successfully ===")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
