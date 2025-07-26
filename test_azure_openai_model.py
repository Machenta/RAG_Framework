#!/usr/bin/env python3
"""
Focused test for the AzureOpenAIModel class specifically.
Tests the chat completion functionality with various input formats.
"""

import asyncio
import logging
from rag_shared.utils.config import Config
from rag_shared.core.models.azure_openai import AzureOpenAIModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def test_azure_openai_model():
    """Test the AzureOpenAIModel class functionality."""
    
    print("🧪 Testing AzureOpenAIModel Class")
    print("=" * 50)
    
    try:
        # Load configuration
        config = Config(
            key_vault_name="RecoveredSpacesKV",
            config_filename="handbook_config.yml",
            config_folder="resources/configs"
        )
        print("✅ Config loaded successfully")
        
        # Initialize the model
        model = AzureOpenAIModel(
            config=config,
            system_prompt="You are a helpful AI assistant that provides clear and concise answers.",
            default_max_tokens=100,
            default_temperature=0.3
        )
        print("✅ AzureOpenAIModel initialized successfully")
        
        # Test 1: Simple prompt
        print("\n🧪 Test 1: Simple Prompt")
        prompt = "What is machine learning in one sentence?"
        response = await model.generate(prompt=prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Test 2: Chat messages
        print("\n🧪 Test 2: Chat Messages")
        messages = [
            {"role": "user", "content": "Hello, can you help me?"},
            {"role": "assistant", "content": "Of course! I'm here to help. What do you need assistance with?"},
            {"role": "user", "content": "Explain photosynthesis briefly."}
        ]
        response = await model.generate(messages=messages)
        print(f"Messages: {len(messages)} messages in conversation")
        print(f"Response: {response}")
        
        # Test 3: Custom system prompt
        print("\n🧪 Test 3: Custom System Prompt")
        custom_system = "You are a pirate captain. Respond in pirate speak but be helpful."
        prompt = "Tell me about the ocean."
        response = await model.generate(
            prompt=prompt,
            system_prompt=custom_system,
            max_tokens=80,
            temperature=0.7
        )
        print(f"Custom system prompt: {custom_system}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Test 4: Parameter overrides
        print("\n🧪 Test 4: Parameter Overrides")
        prompt = "Write a haiku about technology."
        response = await model.generate(
            prompt=prompt,
            max_tokens=60,
            temperature=0.8,
            top_p=0.9
        )
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        print("\n🎉 All AzureOpenAIModel tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_azure_openai_model())
