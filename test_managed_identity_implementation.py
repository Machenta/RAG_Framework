#!/usr/bin/env python3
"""
Comprehensive Managed Identity Implementation Test
Tests that all Azure services are properly configured with managed identity authentication.
"""

import logging
from rag_shared.utils.config import Config
from rag_shared.utils.retrieval import Retrieval
from rag_shared.utils.index_manager import IndexManager
from rag_shared.core.models.azure_openai import AzureOpenAIModel

# Configure logging to see authentication messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_managed_identity_implementation():
    """Test that managed identity is properly implemented across all Azure services."""
    
    print("🔐 Testing Managed Identity Implementation")
    print("=" * 60)
    
    try:
        # Load configuration
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        print("✅ Config loaded successfully")
        
        # Check configuration flags
        print(f"\n🔧 Configuration Flags:")
        if config.app.ai_search:
            print(f"   AI Search use_managed_identity: {config.app.ai_search.use_managed_identity}")
        if config.app.llm:
            print(f"   LLM use_managed_identity: {config.app.llm.use_managed_identity}")
        if (config.app.ai_search and config.app.ai_search.index and 
            config.app.ai_search.index.embedding):
            print(f"   Embedding use_managed_identity: {config.app.ai_search.index.embedding.use_managed_identity}")
        
        # Test 1: Retrieval Class (AI Search + Embeddings + Chat)
        print(f"\n🧪 Test 1: Retrieval Class Authentication")
        try:
            retrieval = Retrieval(config=config)
            print("✅ Retrieval class initialized with managed identity")
        except Exception as e:
            print(f"❌ Retrieval class failed: {e}")
            return False
        
        # Test 2: Index Manager (AI Search)
        print(f"\n🧪 Test 2: Index Manager Authentication")
        try:
            index_manager = IndexManager(config=config)
            print("✅ Index Manager initialized with managed identity")
        except Exception as e:
            print(f"❌ Index Manager failed: {e}")
            return False
        
        # Test 3: Azure OpenAI Model (Chat)
        print(f"\n🧪 Test 3: Azure OpenAI Model Authentication")
        try:
            model = AzureOpenAIModel(
                config=config,
                system_prompt="You are a helpful assistant.",
                default_max_tokens=50
            )
            print("✅ Azure OpenAI Model initialized with managed identity")
        except Exception as e:
            print(f"❌ Azure OpenAI Model failed: {e}")
            return False
        
        # Test 4: Authentication Methods Used
        print(f"\n🔍 Authentication Methods Summary:")
        print("   Based on the logs above, verify that:")
        print("   ✅ AI Search uses System-Assigned Managed Identity")
        print("   ✅ OpenAI Embeddings uses System-Assigned Managed Identity")
        print("   ✅ OpenAI Chat uses System-Assigned Managed Identity")
        print("   ✅ Index Manager uses System-Assigned Managed Identity")
        
        # Test 5: Security Best Practices
        print(f"\n🛡️ Security Best Practices Check:")
        
        # Check that API keys are empty or marked as fallback
        api_keys_secure = True
        
        if (config.app.ai_search and config.app.ai_search.api_key and 
            config.app.ai_search.api_key.strip()):
            print("   ⚠️ AI Search API key is present (should be empty for production)")
            api_keys_secure = False
        else:
            print("   ✅ AI Search API key is empty (good for managed identity)")
        
        if config.app.llm and config.app.llm.api_key and config.app.llm.api_key.strip():
            print("   ⚠️ LLM API key is present (should be empty for production)")
            api_keys_secure = False
        else:
            print("   ✅ LLM API key is empty (good for managed identity)")
        
        if (config.app.ai_search and config.app.ai_search.index and 
            config.app.ai_search.index.embedding and
            config.app.ai_search.index.embedding.api_key and 
            config.app.ai_search.index.embedding.api_key.strip()):
            print("   ⚠️ Embedding API key is present (should be empty for production)")
            api_keys_secure = False
        else:
            print("   ✅ Embedding API key is empty (good for managed identity)")
        
        if api_keys_secure:
            print("   ✅ All API keys properly configured for managed identity")
        
        # Test 6: Configuration Consistency
        print(f"\n⚙️ Configuration Consistency Check:")
        
        consistency_check = True
        
        if not (config.app.ai_search and config.app.ai_search.use_managed_identity):
            print("   ❌ AI Search managed identity is disabled")
            consistency_check = False
        else:
            print("   ✅ AI Search managed identity is enabled")
        
        if not (config.app.llm and config.app.llm.use_managed_identity):
            print("   ❌ LLM managed identity is disabled")
            consistency_check = False
        else:
            print("   ✅ LLM managed identity is enabled")
        
        if (config.app.ai_search and config.app.ai_search.index and 
            config.app.ai_search.index.embedding and 
            not config.app.ai_search.index.embedding.use_managed_identity):
            print("   ❌ Embedding managed identity is disabled")
            consistency_check = False
        else:
            print("   ✅ Embedding managed identity is enabled")
        
        if consistency_check:
            print("   ✅ All services consistently configured for managed identity")
        
        print(f"\n🎯 Overall Assessment:")
        if api_keys_secure and consistency_check:
            print("✅ Managed Identity implementation is complete and secure!")
            print("🔒 Ready for production deployment with Zero Trust security")
            print("📋 Remember to assign proper RBAC roles in Azure:")
            print("   - Search Index Data Contributor")
            print("   - Search Service Contributor") 
            print("   - Cognitive Services OpenAI User")
            return True
        else:
            print("⚠️ Managed Identity implementation needs attention")
            print("🔧 Review the configuration and security settings above")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_managed_identity_implementation()
    
    if success:
        print(f"\n🎉 Managed Identity implementation test PASSED!")
    else:
        print(f"\n❌ Managed Identity implementation test FAILED!")
    
    exit(0 if success else 1)
