from typing import Literal, Dict, Type
from enum import Enum

from core.models.llm import *

from pydantic import ValidationError


SUPPORTED_SOURCES: Dict[str, Type[LLMBaseConfig]] = {
    "openai": OpenAIConfig,
    "azureopenai": AzureOpenAIConfig,
    "ollama": OllamaConfig,
    "groq": GroqConfig,
    "openai-like": OpenAILikeConfig,
}

class OpenAISupportedClients(Enum):
    OPENAI = "openai"
    AOAI = "azureopenai"
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI_LIKE = "openai-like"


def generate_client_config(
    source: Literal["openai", "azureopenai", "ollama", "groq", "openai-like"],
    **kwargs,
) -> LLMBaseConfig:
    """
    生成客户端配置

    Args:
        source (str): 数据源
        **kwargs: 配置参数
    Returns:
        config (LLMBaseConfig): LLM配置
    """
    return SUPPORTED_SOURCES[source].from_env(**kwargs)


def validate_client_config(model_type: str, config: Dict) -> Dict:
    """验证配置并返回标准化的字典"""
    model_class = SUPPORTED_SOURCES[model_type.lower()]

    # 通过 Pydantic 模型验证配置
    try:
        validated_config = model_class.model_validate(config)
        return validated_config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}")


def get_client_config_model(config: Dict) -> LLMConfigType:
    """从配置自动选择模型"""
    model_type = config.get("config_type")
    model_class = SUPPORTED_SOURCES[model_type.lower()]
    
    # 确保配置符合指定类型的要求
    try:
        return model_class.model_validate(config)
    except ValidationError as e:
        raise ValueError(f"Configuration is not valid for {model_type}: {str(e)}")
