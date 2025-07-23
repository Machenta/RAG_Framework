#!/usr/bin/env python3
"""
Quick test script to validate metadata extraction configuration
"""

import yaml
from rag_shared.utils.config import Config
from rag_shared.utils.config_dataclasses import AzureSearchFetcherConfig

def test_metadata_config():
    """Test that the metadata configuration loads correctly"""
    
    # Load the config
    config = Config(
        key_vault_name="RecoveredSpacesKV",
        config_filename="handbook_config.yml",
        config_folder="resources/configs"
    )
    
    # Check if AzureSearchFetcher config exists
    assert config.app is not None, "App config should not be None"
    assert config.app.fetchers is not None, "Fetchers config should not be None"
    assert config.app.fetchers.AzureSearchFetcher is not None, "AzureSearchFetcher config should not be None"
    
    fetcher_config = config.app.fetchers.AzureSearchFetcher
    
    # Check metadata fields
    expected_fields = [
        "filename", "video_url", "timestamp", "chunk_index", 
        "speaker", "topic", "keyword", "questionoranswer", "id"
    ]
    
    print(f"Configured metadata fields: {fetcher_config.metadata_fields}")
    
    if fetcher_config.metadata_fields:
        for field in expected_fields:
            assert field in fetcher_config.metadata_fields, f"Field '{field}' should be in metadata_fields"
    
    # Check select fields in params
    if fetcher_config.params and fetcher_config.params.select_fields:
        select_fields = fetcher_config.params.select_fields
        print(f"Configured select fields: {select_fields}")
        
        # Verify that all metadata fields are also in select fields
        if fetcher_config.metadata_fields:
            for field in fetcher_config.metadata_fields:
                assert field in select_fields, f"Metadata field '{field}' should be in select_fields"
    
    print("✅ All metadata configuration tests passed!")

def test_schema_consistency():
    """Test that configuration matches the transcripts schema"""
    
    # Load the schema
    with open("resources/AI_search_indexes/transcripts.yml", "r") as f:
        schema = yaml.safe_load(f)
    
    # Extract field names from schema
    schema_fields = {field["name"] for field in schema["fields"]}
    print(f"Available schema fields: {sorted(schema_fields)}")
    
    # Load the config
    config = Config(
        key_vault_name="RecoveredSpacesKV",
        config_filename="handbook_config.yml", 
        config_folder="resources/configs"
    )
    
    fetcher_config = config.app.fetchers.AzureSearchFetcher if config.app and config.app.fetchers else None
    
    # Check that all configured fields exist in schema
    if fetcher_config and fetcher_config.metadata_fields:
        for field in fetcher_config.metadata_fields:
            assert field in schema_fields, f"Metadata field '{field}' not found in schema. Available: {sorted(schema_fields)}"
    
    if fetcher_config and fetcher_config.params and fetcher_config.params.select_fields:
        for field in fetcher_config.params.select_fields:
            assert field in schema_fields, f"Select field '{field}' not found in schema. Available: {sorted(schema_fields)}"
    
    print("✅ Schema consistency tests passed!")

if __name__ == "__main__":
    print("Testing metadata extraction configuration...")
    test_metadata_config()
    test_schema_consistency()
    print("\n🎉 All tests passed! Metadata extraction should work correctly.")
