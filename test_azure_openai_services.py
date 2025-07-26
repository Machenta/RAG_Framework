#!/usr/bin/env python3
"""
Comprehensive test script for Azure OpenAI services (Chat and Embeddings)
with System-Assigned Managed Identity authentication.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any
import numpy as np
from rag_shared.utils.config import Config
from rag_shared.core.models.azure_openai import AzureOpenAIModel
from rag_shared.utils.retrieval import Retrieval

# Configure logging to see authentication messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AzureOpenAITester:
    """Test suite for Azure OpenAI Chat and Embeddings services."""
    
    def __init__(self):
        """Initialize the tester with config and clients."""
        print("🔧 Initializing Azure OpenAI Test Suite")
        print("=" * 60)
        
        # Load configuration
        try:
            self.config = Config(
                key_vault_name="RecoveredSpacesKV",
                config_filename="handbook_config.yml",
                config_folder="resources/configs"
            )
            print(f"✅ Config loaded successfully")
            print(f"   App name: {self.config.app.name}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            raise
        
        # Initialize retrieval for embeddings testing
        try:
            self.retrieval = Retrieval(config=self.config)
            print(f"✅ Retrieval initialized for embeddings testing")
        except Exception as e:
            print(f"❌ Failed to initialize retrieval: {e}")
            raise
        
        # Initialize Azure OpenAI model for chat testing
        try:
            self.chat_model = AzureOpenAIModel(
                config=self.config,
                system_prompt="You are a helpful AI assistant specialized in providing clear, concise answers.",
                default_max_tokens=150,
                default_temperature=0.3
            )
            print(f"✅ Azure OpenAI Chat model initialized")
        except Exception as e:
            print(f"❌ Failed to initialize chat model: {e}")
            raise
    
    def test_configuration_display(self):
        """Display the configuration being used for testing."""
        print(f"\n🔧 Configuration Check:")
        
        # LLM Configuration
        if self.config.app.llm:
            print(f"   Chat Model:")
            print(f"     - Deployment: {self.config.app.llm.deployment}")
            print(f"     - API Version: {self.config.app.llm.api_version}")
            print(f"     - Endpoint: {self.config.app.llm.api_base_url}")
            print(f"     - Use Managed Identity: {self.config.app.llm.use_managed_identity}")
        
        # Embeddings Configuration
        if (self.config.app.ai_search and 
            self.config.app.ai_search.index and 
            self.config.app.ai_search.index.embedding):
            embedding_cfg = self.config.app.ai_search.index.embedding
            print(f"   Embeddings Model:")
            print(f"     - Deployment: {embedding_cfg.deployment}")
            print(f"     - API Version: {embedding_cfg.api_version}")
            print(f"     - URL: {embedding_cfg.url}")
            print(f"     - Use Managed Identity: {embedding_cfg.use_managed_identity}")
    
    async def test_chat_completion_simple(self):
        """Test simple chat completion with a direct prompt."""
        print(f"\n🗨️ Testing Chat Completion (Simple Prompt)")
        print("-" * 40)
        
        try:
            test_prompt = "What is the capital of France? Answer in one sentence."
            
            print(f"Sending prompt: '{test_prompt}'")
            
            response = await self.chat_model.generate(prompt=test_prompt)
            
            print(f"✅ Chat completion successful")
            print(f"Response: {response}")
            
            # Validate response
            if response and len(response.strip()) > 0:
                print(f"✅ Response validation passed")
                return True
            else:
                print(f"❌ Response validation failed: empty response")
                return False
                
        except Exception as e:
            print(f"❌ Chat completion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_chat_completion_conversation(self):
        """Test chat completion with conversation history."""
        print(f"\n💬 Testing Chat Completion (Conversation)")
        print("-" * 40)
        
        try:
            # Simulate a conversation
            messages = [
                {"role": "user", "content": "Hello! What's your name?"},
                {"role": "assistant", "content": "Hello! I'm an AI assistant. How can I help you today?"},
                {"role": "user", "content": "Can you tell me a fun fact about space?"}
            ]
            
            print(f"Sending conversation with {len(messages)} messages")
            
            response = await self.chat_model.generate(messages=messages)
            
            print(f"✅ Conversation completion successful")
            print(f"Response: {response}")
            
            # Validate response
            if response and len(response.strip()) > 0:
                print(f"✅ Conversation validation passed")
                return True
            else:
                print(f"❌ Conversation validation failed: empty response")
                return False
                
        except Exception as e:
            print(f"❌ Conversation completion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_chat_completion_with_parameters(self):
        """Test chat completion with custom parameters."""
        print(f"\n⚙️ Testing Chat Completion (Custom Parameters)")
        print("-" * 40)
        
        try:
            test_prompt = "Write a creative short story about a robot learning to paint. Keep it under 100 words."
            
            # Custom parameters for creative response
            custom_params = {
                "max_tokens": 120,
                "temperature": 0.8,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            print(f"Sending prompt with custom parameters:")
            for param, value in custom_params.items():
                print(f"  {param}: {value}")
            
            response = await self.chat_model.generate(
                prompt=test_prompt,
                **custom_params
            )
            
            print(f"✅ Custom parameters completion successful")
            print(f"Response: {response}")
            
            # Validate response
            if response and len(response.strip()) > 0:
                print(f"✅ Custom parameters validation passed")
                return True
            else:
                print(f"❌ Custom parameters validation failed: empty response")
                return False
                
        except Exception as e:
            print(f"❌ Custom parameters completion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_embeddings_single_text(self):
        """Test embeddings generation for a single text."""
        print(f"\n🔢 Testing Embeddings (Single Text)")
        print("-" * 40)
        
        try:
            test_text = "Azure OpenAI provides powerful language models for various AI applications."
            
            # Create a test document
            test_doc = {
                "id": "embedding-test-001",
                self.retrieval._index_cfg.index_text_field: test_text
            }
            
            print(f"Generating embeddings for: '{test_text}'")
            
            result = self.retrieval.embed([test_doc])
            
            if result and len(result) > 0:
                doc_with_embedding = result[0]
                embedding = doc_with_embedding.get(self.retrieval._index_cfg.vector_field)
                
                if embedding and isinstance(embedding, list) and len(embedding) > 0:
                    print(f"✅ Embeddings generation successful")
                    print(f"   Embedding dimensions: {len(embedding)}")
                    print(f"   First 5 values: {embedding[:5]}")
                    
                    # Validate embedding properties
                    if len(embedding) == 1536:  # text-embedding-ada-002 dimension
                        print(f"✅ Embedding dimension validation passed")
                        return True
                    else:
                        print(f"❌ Unexpected embedding dimension: {len(embedding)}")
                        return False
                else:
                    print(f"❌ No valid embedding found in result")
                    return False
            else:
                print(f"❌ No result returned from embeddings")
                return False
                
        except Exception as e:
            print(f"❌ Embeddings generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_embeddings_multiple_texts(self):
        """Test embeddings generation for multiple texts."""
        print(f"\n🔢 Testing Embeddings (Multiple Texts)")
        print("-" * 40)
        
        try:
            test_texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand human language.",
                "Computer vision allows machines to interpret and analyze visual information."
            ]
            
            # Create test documents
            test_docs = []
            for i, text in enumerate(test_texts):
                test_docs.append({
                    "id": f"embedding-test-multi-{i+1:03d}",
                    self.retrieval._index_cfg.index_text_field: text
                })
            
            print(f"Generating embeddings for {len(test_texts)} texts")
            
            result = self.retrieval.embed(test_docs)
            
            if result and len(result) == len(test_texts):
                print(f"✅ Multiple embeddings generation successful")
                
                # Validate each embedding
                all_valid = True
                for i, doc in enumerate(result):
                    embedding = doc.get(self.retrieval._index_cfg.vector_field)
                    if not embedding or len(embedding) != 1536:
                        print(f"❌ Invalid embedding for text {i+1}")
                        all_valid = False
                    else:
                        print(f"   Text {i+1}: {len(embedding)} dimensions")
                
                if all_valid:
                    print(f"✅ All embeddings validation passed")
                    return True
                else:
                    print(f"❌ Some embeddings validation failed")
                    return False
            else:
                print(f"❌ Expected {len(test_texts)} results, got {len(result) if result else 0}")
                return False
                
        except Exception as e:
            print(f"❌ Multiple embeddings generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_embeddings_similarity(self):
        """Test embedding similarity calculation."""
        print(f"\n📊 Testing Embeddings Similarity")
        print("-" * 40)
        
        try:
            # Test texts with different similarity levels
            similar_texts = [
                "Dogs are loyal pets that make great companions.",
                "Canines are faithful animals that serve as excellent friends."
            ]
            
            different_texts = [
                "Dogs are loyal pets that make great companions.",
                "The weather today is sunny with clear blue skies."
            ]
            
            # Generate embeddings for similar texts
            similar_docs = []
            for i, text in enumerate(similar_texts):
                similar_docs.append({
                    "id": f"similarity-similar-{i+1}",
                    self.retrieval._index_cfg.index_text_field: text
                })
            
            similar_result = self.retrieval.embed(similar_docs)
            
            # Generate embeddings for different texts
            different_docs = []
            for i, text in enumerate(different_texts):
                different_docs.append({
                    "id": f"similarity-different-{i+1}",
                    self.retrieval._index_cfg.index_text_field: text
                })
            
            different_result = self.retrieval.embed(different_docs)
            
            if (similar_result and len(similar_result) == 2 and
                different_result and len(different_result) == 2):
                
                # Calculate cosine similarity
                def cosine_similarity(a, b):
                    a = np.array(a)
                    b = np.array(b)
                    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                
                similar_emb1 = similar_result[0][self.retrieval._index_cfg.vector_field]
                similar_emb2 = similar_result[1][self.retrieval._index_cfg.vector_field]
                
                different_emb1 = different_result[0][self.retrieval._index_cfg.vector_field]
                different_emb2 = different_result[1][self.retrieval._index_cfg.vector_field]
                
                similar_similarity = cosine_similarity(similar_emb1, similar_emb2)
                different_similarity = cosine_similarity(different_emb1, different_emb2)
                
                print(f"✅ Similarity calculation successful")
                print(f"   Similar texts similarity: {similar_similarity:.4f}")
                print(f"   Different texts similarity: {different_similarity:.4f}")
                
                # Validate that similar texts have higher similarity
                if similar_similarity > different_similarity:
                    print(f"✅ Similarity validation passed (similar > different)")
                    return True
                else:
                    print(f"❌ Similarity validation failed (similar: {similar_similarity:.4f} <= different: {different_similarity:.4f})")
                    return False
            else:
                print(f"❌ Failed to generate embeddings for similarity test")
                return False
                
        except Exception as e:
            print(f"❌ Similarity test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all tests and provide a summary."""
        print(f"\n🧪 Running Azure OpenAI Services Test Suite")
        print("=" * 60)
        
        # Display configuration
        self.test_configuration_display()
        
        # Run tests
        tests = [
            ("Chat Completion (Simple)", self.test_chat_completion_simple()),
            ("Chat Completion (Conversation)", self.test_chat_completion_conversation()),
            ("Chat Completion (Custom Parameters)", self.test_chat_completion_with_parameters()),
            ("Embeddings (Single Text)", self.test_embeddings_single_text()),
            ("Embeddings (Multiple Texts)", self.test_embeddings_multiple_texts()),
            ("Embeddings Similarity", self.test_embeddings_similarity()),
        ]
        
        results = []
        for test_name, test_coro in tests:
            try:
                if asyncio.iscoroutine(test_coro):
                    result = await test_coro
                else:
                    result = test_coro
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ Test '{test_name}' failed with exception: {e}")
                results.append((test_name, False))
        
        # Print summary
        print(f"\n📊 Test Summary")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} - {test_name}")
            if result:
                passed += 1
        
        print(f"\n🎯 Overall Results: {passed}/{total} tests passed")
        
        if passed == total:
            print(f"🎉 All tests passed! Azure OpenAI services are working correctly.")
            return True
        else:
            print(f"⚠️ Some tests failed. Check the logs above for details.")
            return False

async def main():
    """Main test execution function."""
    try:
        tester = AzureOpenAITester()
        success = await tester.run_all_tests()
        
        if success:
            print(f"\n✅ Azure OpenAI services test completed successfully!")
        else:
            print(f"\n❌ Azure OpenAI services test completed with failures!")
            
        return success
        
    except Exception as e:
        print(f"❌ Test suite failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
