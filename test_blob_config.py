#!/usr/bin/env python3
"""
Test script to validate blob storage config loading functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_shared.utils.config import Config

def test_filesystem_config():
    """Test loading config from filesystem (default behavior)."""
    print("=" * 60)
    print("Testing filesystem config loading...")
    
    try:
        # Make sure we're using filesystem mode
        os.environ.pop("CONFIG_SOURCE", None)
        
        # Test loading the existing config
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        
        print(f"✅ Successfully loaded config from filesystem")
        print(f"   - System prompts count: {len(config.app.llm.prompts.blob_config.system_prompts)}")
        print(f"   - Log level: {config.app.other.log_level if config.app.other else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load filesystem config: {e}")
        return False

def test_blob_config_simulation():
    """Test blob storage config loading setup (without actual storage)."""
    print("=" * 60)
    print("Testing blob storage config setup...")
    
    try:
        # Set environment variables for blob storage mode
        os.environ["CONFIG_SOURCE"] = "blob_storage"
        os.environ["CONFIG_STORAGE_ACCOUNT"] = "teststorage"
        os.environ["CONFIG_CONTAINER"] = "configs"
        os.environ["CONFIG_FILENAME"] = "handbook_config.yml"
        
        print("✅ Environment variables set for blob storage mode:")
        print(f"   - CONFIG_SOURCE: {os.environ.get('CONFIG_SOURCE')}")
        print(f"   - CONFIG_STORAGE_ACCOUNT: {os.environ.get('CONFIG_STORAGE_ACCOUNT')}")
        print(f"   - CONFIG_CONTAINER: {os.environ.get('CONFIG_CONTAINER')}")
        print(f"   - CONFIG_FILENAME: {os.environ.get('CONFIG_FILENAME')}")
        
        # This should fall back to filesystem since blob storage won't be accessible
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        
        print(f"✅ Config loaded with fallback to filesystem")
        print(f"   - System prompts count: {len(config.app.prompts.blob.system_prompts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed blob storage config test: {e}")
        return False
    finally:
        # Clean up environment variables
        for key in ["CONFIG_SOURCE", "CONFIG_STORAGE_ACCOUNT", "CONFIG_CONTAINER", "CONFIG_FILENAME"]:
            os.environ.pop(key, None)

def test_config_validation():
    """Test that the updated config structure is valid."""
    print("=" * 60)
    print("Testing config structure validation...")
    
    try:
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        
        # Test system_prompts list
        system_prompts = config.app.prompts.blob.system_prompts
        print(f"✅ System prompts loaded: {system_prompts}")
        
        # Test that it's a list (not a string)
        assert isinstance(system_prompts, list), f"Expected list, got {type(system_prompts)}"
        assert len(system_prompts) > 0, "System prompts list should not be empty"
        
        print(f"✅ Config structure validation passed")
        print(f"   - System prompts is list: {isinstance(system_prompts, list)}")
        print(f"   - System prompts count: {len(system_prompts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting RAG Framework Config Tests")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    tests = [
        test_filesystem_config,
        test_config_validation,
        test_blob_config_simulation,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test.__name__}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
