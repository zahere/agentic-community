"""
Multi-LLM Provider Support

This module provides a unified interface for multiple LLM providers including
OpenAI, Anthropic, Google, Mistral, and more.
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json

from agentic_community.core.exceptions import ProviderError
from agentic_community.core.cache import cache_result
from agentic_community.core.validation import validate_input


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider"""
    provider: LLMProvider
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider-specific configuration"""
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion from messages"""
        pass
    
    @abstractmethod
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion responses"""
        pass
    
    def _merge_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided kwargs with config defaults"""
        merged = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        merged.update(self.config.extra_params)
        merged.update(kwargs)
        return merged


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-3.5, GPT-4, etc.)"""
    
    def _validate_config(self) -> None:
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        if not self.config.api_key:
            raise ProviderError("OpenAI API key not provided")
        
        if not self.config.model:
            self.config.model = "gpt-4"
        
        if not self.config.api_base:
            self.config.api_base = "https://api.openai.com/v1"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API"""
        params = self._merge_kwargs(kwargs)
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": params.get("model", self.config.model),
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
            "frequency_penalty": params.get("frequency_penalty", self.config.frequency_penalty),
            "presence_penalty": params.get("presence_penalty", self.config.presence_penalty),
        }
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.retry_attempts):
                try:
                    async with session.post(
                        f"{self.config.api_base}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        response_data = await response.json()
                        
                        if response.status != 200:
                            raise ProviderError(f"OpenAI API error: {response_data}")
                        
                        return LLMResponse(
                            content=response_data["choices"][0]["message"]["content"],
                            model=response_data["model"],
                            provider=LLMProvider.OPENAI,
                            usage=response_data.get("usage", {}),
                            raw_response=response_data
                        )
                        
                except asyncio.TimeoutError:
                    if attempt == self.config.retry_attempts - 1:
                        raise ProviderError("OpenAI API timeout")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise ProviderError(f"OpenAI API error: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
    
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion from OpenAI"""
        params = self._merge_kwargs(kwargs)
        queue = asyncio.Queue()
        
        # Implementation would handle SSE streaming
        # Placeholder for now
        await queue.put({"content": "Streaming not implemented yet"})
        await queue.put(None)  # Signal end
        
        return queue


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude, Claude 2, etc.)"""
    
    def _validate_config(self) -> None:
        if not self.config.api_key:
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.config.api_key:
            raise ProviderError("Anthropic API key not provided")
        
        if not self.config.model:
            self.config.model = "claude-3-opus-20240229"
        
        if not self.config.api_base:
            self.config.api_base = "https://api.anthropic.com/v1"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Anthropic API"""
        params = self._merge_kwargs(kwargs)
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Convert messages to Anthropic format
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        data = {
            "model": params.get("model", self.config.model),
            "messages": anthropic_messages,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
        }
        
        if system_message:
            data["system"] = system_message
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.config.retry_attempts):
                try:
                    async with session.post(
                        f"{self.config.api_base}/messages",
                        headers=headers,
                        json=data,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        response_data = await response.json()
                        
                        if response.status != 200:
                            raise ProviderError(f"Anthropic API error: {response_data}")
                        
                        return LLMResponse(
                            content=response_data["content"][0]["text"],
                            model=response_data["model"],
                            provider=LLMProvider.ANTHROPIC,
                            usage={
                                "prompt_tokens": response_data["usage"]["input_tokens"],
                                "completion_tokens": response_data["usage"]["output_tokens"],
                                "total_tokens": response_data["usage"]["input_tokens"] + 
                                               response_data["usage"]["output_tokens"]
                            },
                            raw_response=response_data
                        )
                        
                except asyncio.TimeoutError:
                    if attempt == self.config.retry_attempts - 1:
                        raise ProviderError("Anthropic API timeout")
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise ProviderError(f"Anthropic API error: {str(e)}")
                    await asyncio.sleep(2 ** attempt)
    
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion from Anthropic"""
        # Similar to OpenAI, would implement SSE streaming
        queue = asyncio.Queue()
        await queue.put({"content": "Streaming not implemented yet"})
        await queue.put(None)
        return queue


class GoogleProvider(BaseLLMProvider):
    """Google AI provider (Gemini, PaLM, etc.)"""
    
    def _validate_config(self) -> None:
        if not self.config.api_key:
            self.config.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.config.api_key:
            raise ProviderError("Google API key not provided")
        
        if not self.config.model:
            self.config.model = "gemini-pro"
        
        if not self.config.api_base:
            self.config.api_base = "https://generativelanguage.googleapis.com/v1beta"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Google AI API"""
        params = self._merge_kwargs(kwargs)
        
        # Convert messages to Google format
        contents = []
        for msg in messages:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": params["temperature"],
                "topP": params["top_p"],
                "maxOutputTokens": params["max_tokens"],
            }
        }
        
        url = f"{self.config.api_base}/models/{self.config.model}:generateContent?key={self.config.api_key}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise ProviderError(f"Google AI API error: {response_data}")
                
                content = response_data["candidates"][0]["content"]["parts"][0]["text"]
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider=LLMProvider.GOOGLE,
                    usage=response_data.get("usageMetadata", {}),
                    raw_response=response_data
                )
    
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion from Google AI"""
        queue = asyncio.Queue()
        await queue.put({"content": "Streaming not implemented yet"})
        await queue.put(None)
        return queue


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider"""
    
    def _validate_config(self) -> None:
        if not self.config.api_key:
            self.config.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.config.api_key:
            raise ProviderError("Mistral API key not provided")
        
        if not self.config.model:
            self.config.model = "mistral-large-latest"
        
        if not self.config.api_base:
            self.config.api_base = "https://api.mistral.ai/v1"
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Mistral API"""
        params = self._merge_kwargs(kwargs)
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": params.get("model", self.config.model),
            "messages": messages,
            "temperature": params["temperature"],
            "max_tokens": params["max_tokens"],
            "top_p": params["top_p"],
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response_data = await response.json()
                
                if response.status != 200:
                    raise ProviderError(f"Mistral API error: {response_data}")
                
                return LLMResponse(
                    content=response_data["choices"][0]["message"]["content"],
                    model=response_data["model"],
                    provider=LLMProvider.MISTRAL,
                    usage=response_data.get("usage", {}),
                    raw_response=response_data
                )
    
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion from Mistral"""
        queue = asyncio.Queue()
        await queue.put({"content": "Streaming not implemented yet"})
        await queue.put(None)
        return queue


class LLMProviderFactory:
    """Factory for creating LLM provider instances"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.GOOGLE: GoogleProvider,
        LLMProvider.MISTRAL: MistralProvider,
    }
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        """Create an LLM provider instance"""
        provider_class = cls._providers.get(config.provider)
        
        if not provider_class:
            raise ProviderError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def register_provider(
        cls,
        provider: LLMProvider,
        provider_class: type
    ) -> None:
        """Register a custom provider"""
        cls._providers[provider] = provider_class


class MultiLLMClient:
    """
    Unified client for multiple LLM providers with fallback support
    """
    
    def __init__(
        self,
        primary_config: LLMConfig,
        fallback_configs: Optional[List[LLMConfig]] = None
    ):
        self.primary_provider = LLMProviderFactory.create(primary_config)
        self.fallback_providers = [
            LLMProviderFactory.create(config)
            for config in (fallback_configs or [])
        ]
        
    @cache_result(ttl=3600)
    async def complete(
        self,
        messages: List[Dict[str, str]],
        use_fallback: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback
        
        Args:
            messages: List of message dictionaries
            use_fallback: Whether to use fallback providers on failure
            **kwargs: Additional parameters for the provider
            
        Returns:
            LLMResponse object
        """
        providers = [self.primary_provider]
        if use_fallback:
            providers.extend(self.fallback_providers)
        
        last_error = None
        for provider in providers:
            try:
                return await provider.complete(messages, **kwargs)
            except Exception as e:
                last_error = e
                continue
        
        raise ProviderError(f"All providers failed. Last error: {last_error}")
    
    async def complete_with_all(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> List[LLMResponse]:
        """
        Get completions from all configured providers in parallel
        
        Useful for comparing responses or ensemble approaches
        """
        providers = [self.primary_provider] + self.fallback_providers
        
        tasks = [
            provider.complete(messages, **kwargs)
            for provider in providers
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, LLMResponse):
                valid_results.append(result)
            
        return valid_results
    
    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> asyncio.Queue:
        """Stream completion from primary provider"""
        return await self.primary_provider.stream_complete(messages, **kwargs)


# Convenience functions
def create_llm_client(
    provider: Union[str, LLMProvider],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> MultiLLMClient:
    """
    Create an LLM client with a single line
    
    Example:
        client = create_llm_client("openai", model="gpt-4")
        response = await client.complete([{"role": "user", "content": "Hello!"}])
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    
    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        model=model or "",
        **kwargs
    )
    
    return MultiLLMClient(config)


def create_multi_llm_client(
    configs: List[Dict[str, Any]]
) -> MultiLLMClient:
    """
    Create a multi-LLM client with fallback support
    
    Example:
        client = create_multi_llm_client([
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "anthropic", "model": "claude-2"},
            {"provider": "google", "model": "gemini-pro"}
        ])
    """
    if not configs:
        raise ValueError("At least one configuration required")
    
    llm_configs = []
    for config_dict in configs:
        provider = config_dict.pop("provider")
        if isinstance(provider, str):
            provider = LLMProvider(provider)
        
        llm_configs.append(LLMConfig(provider=provider, **config_dict))
    
    return MultiLLMClient(
        primary_config=llm_configs[0],
        fallback_configs=llm_configs[1:] if len(llm_configs) > 1 else None
    )
