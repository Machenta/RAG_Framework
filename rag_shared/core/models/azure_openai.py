import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import openai
from azure.identity import DefaultAzureCredential
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
)
from .base import LLMModel
from rag_shared.utils.config import Config

class AzureOpenAIModel(LLMModel):
    def __init__(
        self,
        config: Config,
        * ,
        system_prompt: str = "You are a helpful assistant.",
        default_max_tokens: int = 800,
        default_temperature: float = 1.0,
        default_top_p: float = 1.0,
        default_frequency_penalty: float = 0.0,
        default_presence_penalty: float = 0.0
    ):
        """
        system_prompt: global instructions for the assistant
        """
        super().__init__()
        
        self.config : Config = config

        self.system_prompt : str = system_prompt

        # Assert the class attributes and not the config attributes
        assert self.config.app is not None, "Config.app is required"
        assert self.config.app.llm is not None, "LLM config is required in config.app.llm"
        assert self.config.app.llm.api_version, "LLM api_version must be provided"
        assert self.config.app.llm.deployment, "LLM deployment must be provided"
        assert self.config.app.llm.api_base_url, "LLM api_base_url must be provided"

        # Initialize Azure OpenAI client with managed identity support
        self.client = self._get_openai_client(self.config.app.llm)
        self.defaults = {
            "max_tokens":        default_max_tokens,
            "temperature":       default_temperature,
            "top_p":             default_top_p,
            "frequency_penalty": default_frequency_penalty,
            "presence_penalty":  default_presence_penalty
        }

        

    def _get_openai_client(self, llm_cfg):
        """
        Get Azure OpenAI client using Managed Identity as primary method.
        Falls back to API key if managed identity is not available or fails.
        """
        use_managed_identity = getattr(llm_cfg, 'use_managed_identity', True)
        
        if use_managed_identity:
            try:
                credential = DefaultAzureCredential()
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                if token:
                    logging.info("✅ Using System-Assigned Managed Identity for Azure OpenAI Chat")
                    return openai.AzureOpenAI(
                        azure_ad_token_provider=lambda: credential.get_token("https://cognitiveservices.azure.com/.default").token,
                        api_version=llm_cfg.api_version,
                        azure_endpoint=llm_cfg.api_base_url,
                        azure_deployment=llm_cfg.deployment
                    )
            except Exception as e:
                logging.warning(f"⚠️ Managed Identity failed for Azure OpenAI Chat: {e}")
                logging.info("🔄 Falling back to API key authentication")
        
        # Fallback to API key
        if hasattr(llm_cfg, 'api_key') and llm_cfg.api_key:
            logging.info("🔑 Using API key for Azure OpenAI Chat authentication")
            return openai.AzureOpenAI(
                api_key=llm_cfg.api_key,
                api_version=llm_cfg.api_version,
                azure_endpoint=llm_cfg.api_base_url,
                azure_deployment=llm_cfg.deployment
            )
        
        raise ValueError("No valid authentication method available for Azure OpenAI Chat. "
                        "Ensure either Managed Identity is configured or API key is provided.")

    async def generate(
        self,
        prompt: Optional[str]                             = None,
        messages: Optional[List[Dict[str, str]]]          = None,
        system_prompt: Optional[str]                      = None,
        **override_kwargs: Any
    ) -> str:
        """
        - prompt: raw user‐only string
        - messages: full chat history as role/content dicts
        - system_prompt: per‐call override of self.system_prompt
        """
        # choose which system prompt to use
        sys_text = system_prompt or self.system_prompt

        # Build the parameterized ChatCompletionMessageParam list
        chat_msgs: List[ChatCompletionMessageParam]
        if messages is None:
            if prompt is None:
                raise ValueError("You must supply either `prompt` or `messages`.")
            chat_msgs = [
                ChatCompletionSystemMessageParam(role="system",  content=sys_text),
                ChatCompletionUserMessageParam(  role="user",    content=prompt)
            ]
        else:
            # Assume the caller may want to inject multiple messages
            chat_msgs = [ ChatCompletionSystemMessageParam(role="system", content=sys_text) ]
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    # if they explicitly include system in messages, append its content too
                    chat_msgs.append(ChatCompletionSystemMessageParam(role="system", content=content))
                elif role == "user":
                    chat_msgs.append(ChatCompletionUserMessageParam(role="user", content=content))
                elif role == "assistant":
                    chat_msgs.append(ChatCompletionAssistantMessageParam(role="assistant", content=content))
                else:
                    raise ValueError(f"Unsupported role: {role}")

        # Merge defaults and overrides
        params = { **self.defaults, **override_kwargs }
    
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model    = self.config.app.llm.deployment, # type: ignore
            messages = chat_msgs,
            **params
        )

        text = response.choices[0].message.content
        return text.strip() if text is not None else ""


if __name__ == "__main__":
    import os
    import asyncio
    from rag_shared.utils.config import Config

    cfg = Config(key_vault_name="RecoveredSpacesKV", config_filename="recovered_config.yml", config_folder="resources/configs")
    system_prompt = open(os.path.join("resources/prompts", "default_prompt_with_json.j2")).read()
    # 2) Instantiate the model
    model = AzureOpenAIModel(cfg,
                             system_prompt=system_prompt,
                             default_temperature=0.5,
                             default_max_tokens=100)

    # 3) Define a test prompt
    test_prompt = "Write a brief poem about the sea."

    # 4) Run it in an asyncio loop
    async def main():
        print("Sending prompt to Azure OpenAI:")
        print("  ", test_prompt, "\n")
        response = await model.generate(test_prompt)
        print("Model response:")
        print(response)

    asyncio.run(main())