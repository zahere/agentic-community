"""
Multi-LLM Provider Support for Agentic Framework.

Provides a unified interface for multiple LLM providers including:
- OpenAI (GPT-4, GPT-4 Turbo)
- Anthropic (Claude 3 Opus, Sonnet, Haiku)
- Google (Gemini Pro, Gemini Ultra)
- Mistral AI (Mistral Large, Medium)
- Cohere (Command R+)
- Local models (Llama 3, Mixtral)
"""

from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    stop_sequences: List[str] = None
    streaming: bool = False
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class LLMResponse:
    """Unified response from LLM providers."""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int]
    metadata: Dict[str, Any] = None
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion."""
        pass
        
    @abstractmethod
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a completion."""
        pass
        
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion."""
        pass
        
    @abstractmethod
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream a chat completion."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion using OpenAI."""
        response = await self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stop=self.config.stop_sequences,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].text,
            model=response.model,
            provider=LLMProvider.OPENAI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a completion from OpenAI."""
        stream = await self.client.completions.create(
            model=self.config.model_name,
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].text:
                yield chunk.choices[0].text
                
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion using OpenAI."""
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stop=self.config.stop_sequences,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=LLMProvider.OPENAI,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream a chat completion from OpenAI."""
        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion using Anthropic."""
        # Anthropic uses messages API
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
        
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a completion from Anthropic."""
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(messages, **kwargs):
            yield chunk
            
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion using Anthropic."""
        response = await self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop_sequences=self.config.stop_sequences,
            **kwargs
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider=LLMProvider.ANTHROPIC,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
        
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream a chat completion from Anthropic."""
        async with self.client.messages.stream(
            model=self.config.model_name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            self.model = genai.GenerativeModel(config.model_name)
        except ImportError:
            raise ImportError("Google AI package not installed. Install with: pip install google-generativeai")
            
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion using Google."""
        response = await self.model.generate_content_async(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "stop_sequences": self.config.stop_sequences
            }
        )
        
        return LLMResponse(
            content=response.text,
            model=self.config.model_name,
            provider=LLMProvider.GOOGLE,
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        )
        
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a completion from Google."""
        response = await self.model.generate_content_async(
            prompt,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text
                
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion using Google."""
        # Convert messages to Google format
        chat = self.model.start_chat()
        
        for message in messages[:-1]:
            if message["role"] == "user":
                chat.send_message(message["content"])
                
        # Send final message
        response = await chat.send_message_async(
            messages[-1]["content"],
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
        )
        
        return LLMResponse(
            content=response.text,
            model=self.config.model_name,
            provider=LLMProvider.GOOGLE,
            usage={"total_tokens": 0}  # Google doesn't provide token counts
        )
        
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream a chat completion from Google."""
        chat = self.model.start_chat()
        
        for message in messages[:-1]:
            if message["role"] == "user":
                chat.send_message(message["content"])
                
        response = await chat.send_message_async(
            messages[-1]["content"],
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from mistralai.async_client import MistralAsyncClient
            self.client = MistralAsyncClient(api_key=config.api_key)
        except ImportError:
            raise ImportError("Mistral package not installed. Install with: pip install mistralai")
            
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a completion using Mistral."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
        
    async def stream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a completion from Mistral."""
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.stream_chat(messages, **kwargs):
            yield chunk
            
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate a chat completion using Mistral."""
        response = await self.client.chat(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=LLMProvider.MISTRAL,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
    async def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[str, None]:
        """Stream a chat completion from Mistral."""
        async_response = self.client.chat_stream(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        
        async for chunk in async_response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMRouter:
    """Routes requests to appropriate LLM providers with fallback support."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.primary_provider: Optional[LLMProvider] = None
        self.fallback_chain: List[LLMProvider] = []
        
    def add_provider(self, config: LLMConfig) -> None:
        """Add a provider to the router."""
        provider_class = {
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
            LLMProvider.GOOGLE: GoogleProvider,
            LLMProvider.MISTRAL: MistralProvider,
        }.get(config.provider)
        
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
            
        self.providers[config.provider] = provider_class(config)
        
        if not self.primary_provider:
            self.primary_provider = config.provider
            
    def set_fallback_chain(self, providers: List[LLMProvider]) -> None:
        """Set the fallback chain for providers."""
        self.fallback_chain = providers
        
    async def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Route completion request with fallback support."""
        providers_to_try = [self.primary_provider] + self.fallback_chain
        
        for provider_type in providers_to_try:
            if provider_type not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_type]
                return await provider.complete(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Provider {provider_type} failed: {e}")
                continue
                
        raise Exception("All providers failed")
        
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Route chat request with fallback support."""
        providers_to_try = [self.primary_provider] + self.fallback_chain
        
        for provider_type in providers_to_try:
            if provider_type not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_type]
                return await provider.chat(messages, **kwargs)
            except Exception as e:
                logger.error(f"Provider {provider_type} failed: {e}")
                continue
                
        raise Exception("All providers failed")


# Convenience functions
def create_llm_config(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMConfig:
    """Create an LLM configuration."""
    return LLMConfig(
        provider=LLMProvider(provider),
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )


async def create_llm_router(configs: List[LLMConfig]) -> LLMRouter:
    """Create an LLM router with multiple providers."""
    router = LLMRouter()
    
    for config in configs:
        router.add_provider(config)
        
    return router
