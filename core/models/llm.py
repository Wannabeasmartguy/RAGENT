import os
from typing import TypeVar
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class LLMParams(BaseModel):
    """LLM 参数配置"""

    temperature: float = Field(
        default=0.5, ge=0, le=1, description="Sampling temperature"
    )
    top_p: float = Field(
        default=1.0, ge=0, le=1, description="Nucleus sampling threshold"
    )
    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum number of tokens to generate"
    )
    stream: bool = Field(
        default=False, description="If true, the LLM will stream the response"
    )


class LLMBaseConfig(BaseModel, ABC):
    """LLM 配置"""

    model: str = Field(..., description="Model name")
    params: LLMParams = Field(default_factory=LLMParams, description="LLM parameters")

    @classmethod
    @abstractmethod
    def from_env(cls, **kwargs) -> "LLMBaseConfig":
        """从环境变量和kwargs创建配置"""
        pass


LLMConfigType = TypeVar("LLMConfigType", bound=LLMBaseConfig)


class AzureOpenAIConfig(LLMBaseConfig):
    """Azure OpenAI配置"""

    api_key: str = Field(..., description="API key")
    base_url: str = Field(..., description="API endpoint")
    api_type: str = Field(default="azure", description="API type")
    api_version: str = Field(default="2024-02-15-preview", description="API version")
    params: LLMParams = Field(default_factory=LLMParams, description="LLM parameters")

    @classmethod
    def from_env(cls, **kwargs) -> "AzureOpenAIConfig":
        """从环境变量和kwargs创建配置"""
        return cls(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            api_key=os.getenv("AZURE_OAI_KEY", kwargs.get("api_key", "noaoaikey")),
            base_url=os.getenv(
                "AZURE_OAI_ENDPOINT", kwargs.get("base_url", "noaoaiendpoint")
            ),
            api_type=os.getenv("API_TYPE", kwargs.get("api_type", "azure")),
            api_version=os.getenv(
                "API_VERSION", kwargs.get("api_version", "2024-02-15-preview")
            ),
            params=LLMParams(
                temperature=kwargs.get("temperature", 0.5),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=kwargs.get("stream", False),
            ),
        )


class OpenAIConfig(LLMBaseConfig):
    """OpenAI配置"""

    api_key: str = Field(..., description="API key")
    base_url: str = Field(..., description="API endpoint")
    params: LLMParams = Field(default_factory=LLMParams, description="LLM parameters")

    @classmethod
    def from_env(cls, **kwargs) -> "OpenAIConfig":
        """从环境变量和kwargs创建配置"""
        return cls(
            model=kwargs.get("model", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY", kwargs.get("api_key", "noopenaikey")),
            base_url=os.getenv(
                "OPENAI_API_ENDPOINT", kwargs.get("base_url", "noopenaiendpoint")
            ),
            params=LLMParams(
                temperature=kwargs.get("temperature", 0.5),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=kwargs.get("stream", False),
            ),
        )


class OllamaConfig(LLMBaseConfig):
    """Ollama配置"""

    api_key: str = Field(..., description="API key, required but not used")
    base_url: str = Field(..., description="API endpoint")
    params: LLMParams = Field(default_factory=LLMParams, description="LLM parameters")

    @classmethod
    def from_env(cls, **kwargs) -> "OllamaConfig":
        """从环境变量和kwargs创建配置"""
        return cls(
            model=kwargs.get("model", "nogiven"),
            api_key=os.getenv("OLLAMA_API_KEY", kwargs.get("api_key", "noollamakey")),
            base_url=os.getenv(
                "OLLAMA_API_ENDPOINT",
                kwargs.get("base_url", "http://localhost:11434/v1"),
            ),
            params=LLMParams(
                temperature=kwargs.get("temperature", 0.5),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=kwargs.get("stream", False),
            ),
        )


class GroqConfig(LLMBaseConfig):
    """Groq配置"""

    api_key: str = Field(..., description="API key")
    base_url: str = Field(..., description="API endpoint")
    params: LLMParams = Field(default_factory=LLMParams, description="LLM parameters")

    @classmethod
    def from_env(cls, **kwargs) -> "GroqConfig":
        """从环境变量和kwargs创建配置"""
        return cls(
            model=kwargs.get("model", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY", kwargs.get("api_key", "nogroqkey")),
            base_url=os.getenv(
                "GROQ_API_ENDPOINT",
                kwargs.get("base_url", "https://api.groq.com/openai/v1"),
            ),
            params=LLMParams(
                temperature=kwargs.get("temperature", 0.5),
                top_p=kwargs.get("top_p", 1.0),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=kwargs.get("stream", False),
            ),
        )
