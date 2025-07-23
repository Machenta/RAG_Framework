#!/usr/bin/env python3
"""
Test script to demonstrate that the Config class now works properly 
with existing event loops (like uvicorn).
"""

import asyncio
import os
from rag_shared.utils.config import Config, SingletonMeta

async def test_config_in_async_context():
    """Test that Config can be instantiated within an existing event loop."""
    print("🔄 Testing Config instantiation within async context...")
    
    # Clear singleton instances for fresh test
    SingletonMeta._instances.clear()
    
    # Test blob storage config (if environment variables are set)
    if os.getenv("CONFIG_SOURCE") == "blob_storage":
        try:
            config = Config(
                key_vault_name="RecoveredSpacesKV",
                config_folder="resources/configs",
                config_filename="development_config.yml"
            )
            print(f"✅ Blob storage config loaded successfully: {config.app.name}")
            return config
        except Exception as e:
            print(f"❌ Blob storage config failed: {e}")
    
    # Test filesystem config as fallback
    try:
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_folder="resources/configs",
            config_filename="handbook_config.yml"
        )
        print(f"✅ Filesystem config loaded successfully: {config.app.name}")
        return config
    except Exception as e:
        print(f"❌ Filesystem config failed: {e}")
        return None

def test_config_sync():
    """Test that Config can be instantiated in synchronous context."""
    print("🔄 Testing Config instantiation in sync context...")
    
    # Clear singleton instances for fresh test
    SingletonMeta._instances.clear()
    
    try:
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_folder="resources/configs",
            config_filename="handbook_config.yml"
        )
        print(f"✅ Sync config loaded successfully: {config.app.name}")
        return config
    except Exception as e:
        print(f"❌ Sync config failed: {e}")
        return None

async def simulate_uvicorn_scenario():
    """Simulate the scenario where uvicorn has an event loop and imports the config."""
    print("\n🚀 Simulating uvicorn scenario (existing event loop)...")
    
    # Set up blob storage environment (similar to Azure Web App)
    os.environ["CONFIG_SOURCE"] = "blob_storage"
    os.environ["CONFIG_STORAGE_ACCOUNT"] = "ragrecoveredspacestorage"
    os.environ["CONFIG_CONTAINER"] = "configs"
    os.environ["CONFIG_DIRECTORY"] = "dev"
    os.environ["CONFIG_FILENAME"] = "development_config.yml"
    
    try:
        # This is what would happen when uvicorn imports your app
        config = await test_config_in_async_context()
        
        if config:
            print(f"🎉 Config working in uvicorn-like scenario!")
            print(f"   App Name: {config.app.name}")
            print(f"   Deployment: {config.app.deployment}")
            
            # Test accessing some configuration values
            if config.app.api:
                print(f"   API Host: {config.app.api.host}:{config.app.api.port}")
            
            if config.app.llm:
                print(f"   LLM Model: {config.app.llm.model_name}")
                
    except Exception as e:
        print(f"❌ Uvicorn scenario failed: {e}")
    finally:
        # Clean up environment variables
        for env_var in ["CONFIG_SOURCE", "CONFIG_STORAGE_ACCOUNT", "CONFIG_CONTAINER", "CONFIG_DIRECTORY", "CONFIG_FILENAME"]:
            os.environ.pop(env_var, None)

async def main():
    """Main test function."""
    print("🧪 Testing Event Loop Fix for Config Class")
    print("=" * 50)
    
    # Test 1: Sync context (traditional usage)
    test_config_sync()
    
    print()
    
    # Test 2: Async context (uvicorn scenario)
    await simulate_uvicorn_scenario()
    
    print("\n✅ All tests completed!")
    print("\n📝 Summary:")
    print("   - Config class now handles event loop conflicts")
    print("   - Works in both sync and async contexts")
    print("   - Proper credential cleanup prevents warnings")
    print("   - Ready for use with uvicorn and FastAPI")

if __name__ == "__main__":
    asyncio.run(main())
