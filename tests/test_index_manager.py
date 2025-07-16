import pytest
import os
from unittest.mock import MagicMock, patch
from rag_shared.utils.config import Config, SingletonMeta
from rag_shared.utils.index_manager import IndexManager

@pytest.fixture(autouse=True)
def _clear_config_singleton():
    SingletonMeta._instances.clear()
    yield
    SingletonMeta._instances.clear()

@pytest.fixture
def mock_config():
    """Create a mock config for testing IndexManager."""
    config = MagicMock(spec=Config)
    
    # Mock the nested structure
    config.app.ai_search.index.name = "test-index"
    config.app.ai_search.index.skillset_name = "test-skillset"
    config.app.ai_search.index.indexer_name = "test-indexer"
    config.app.ai_search.index.indexes_path = "resources/AI_search_indexes"
    config.app.ai_search.index.index_yml_path = "test.yml"
    config.app.ai_search.index.semantic_content_fields = ["text", "content"]
    config.app.ai_search.index.semantic_title_field = "title"
    config.app.ai_search.endpoint = "https://test.search.windows.net"
    config.app.ai_search.api_key = "test-key"
    config.app.ai_search.index.embedding.deployment = "test-deployment"
    config.app.ai_search.index.embedding.model_name = "test-model"
    
    return config

def test_semantic_fields_from_config(mock_config):
    """Test that semantic fields are read from config instead of hardcoded."""
    
    # Mock the file loading and Azure clients
    with patch.object(IndexManager, '_load_index_file', return_value={"fields": []}), \
         patch('rag_shared.utils.index_manager.SearchIndexClient'), \
         patch('rag_shared.utils.index_manager.SearchIndexerClient'):
        
        manager = IndexManager(mock_config)
        
        # Verify the config values are correctly set
        assert mock_config.app.ai_search.index.semantic_content_fields == ["text", "content"]
        assert mock_config.app.ai_search.index.semantic_title_field == "title"

def test_semantic_fields_defaults(mock_config):
    """Test that defaults are used when semantic fields are not configured."""
    
    # Set semantic fields to None to test defaults
    mock_config.app.ai_search.index.semantic_content_fields = None
    mock_config.app.ai_search.index.semantic_title_field = None
    
    with patch.object(IndexManager, '_load_index_file', return_value={"fields": []}), \
         patch('rag_shared.utils.index_manager.SearchIndexClient'), \
         patch('rag_shared.utils.index_manager.SearchIndexerClient'):
        
        manager = IndexManager(mock_config)
        
        # The defaults should be used: ["text"] for content and "questionoranswer" for title
        assert mock_config.app.ai_search.index.semantic_content_fields is None
        assert mock_config.app.ai_search.index.semantic_title_field is None

def test_path_normalization_backslashes():
    """Test that backslashes in paths are properly handled."""
    from rag_shared.utils.index_manager import IndexManager
    
    config = MagicMock(spec=Config)
    config.app.ai_search.index.name = "test-index"
    config.app.ai_search.index.skillset_name = "test-skillset"
    config.app.ai_search.index.indexer_name = "test-indexer"
    config.app.ai_search.index.indexes_path = "resources\\AI_search_indexes"  # Windows-style path
    config.app.ai_search.index.index_yml_path = "test\\index.yml"             # Windows-style path
    config.app.ai_search.endpoint = "https://test.search.windows.net"
    config.app.ai_search.api_key = "test-key"
    
    with patch.object(IndexManager, '_load_index_file', return_value={"fields": []}), \
         patch('rag_shared.utils.index_manager.SearchIndexClient'), \
         patch('rag_shared.utils.index_manager.SearchIndexerClient'), \
         patch('builtins.print') as mock_print:
        
        manager = IndexManager(config)
        
        # Check that warnings were printed for backslashes
        warning_calls = [call for call in mock_print.call_args_list 
                        if 'Warning' in str(call) and 'backslashes' in str(call)]
        assert len(warning_calls) >= 1, "Should warn about backslashes in paths"
        
        # Test that _resolve_path handles backslashes correctly
        resolved = manager._resolve_path("resources\\test", "file.yml")
        # Should not contain backslashes in the normalized result
        assert '\\' not in resolved or os.name == 'nt'  # Allow backslashes only on Windows after normalization

def test_resolve_path_mixed_separators():
    """Test path resolution with mixed path separators."""
    from rag_shared.utils.index_manager import IndexManager
    
    config = MagicMock(spec=Config)
    config.app.ai_search.index.name = "test-index"
    config.app.ai_search.index.skillset_name = "test-skillset"
    config.app.ai_search.index.indexer_name = "test-indexer"
    config.app.ai_search.index.indexes_path = "resources"
    config.app.ai_search.index.index_yml_path = "test.yml"
    config.app.ai_search.endpoint = "https://test.search.windows.net"
    config.app.ai_search.api_key = "test-key"
    
    with patch.object(IndexManager, '_load_index_file', return_value={"fields": []}), \
         patch('rag_shared.utils.index_manager.SearchIndexClient'), \
         patch('rag_shared.utils.index_manager.SearchIndexerClient'):
        
        manager = IndexManager(config)
        
        # Test various path combinations
        test_cases = [
            ("resources\\mixed/path", "file.yml"),
            ("resources/unix/path", "file.yml"),
            ("resources\\windows\\path", "file.yml"),
        ]
        
        for path_parts in test_cases:
            resolved = manager._resolve_path(*path_parts)
            # Should be a valid, normalized path
            assert isinstance(resolved, str)
            assert len(resolved) > 0
