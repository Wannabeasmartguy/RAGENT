from typing import Literal, Dict, Type
from enum import Enum

from core.models.llm import *

from pydantic import ValidationError


SUPPORTED_SOURCES: Dict[str, Type[LLMBaseConfig]] = {
    "openai": OpenAIConfig,
    "aoai": AzureOpenAIConfig,
    "ollama": OllamaConfig,
    "groq": GroqConfig,
    "openai-like": OpenAILikeConfig,
}

class OpenAISupportedClients(Enum):
    OPENAI = "openai"
    AOAI = "aoai"
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI_LIKE = "openai-like"


def generate_client_config(
    source: Literal["openai", "aoai", "llamafile", "ollama", "groq", "openai-like"],
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


def get_client_config(config: Dict) -> str:
    """从配置中获取标准化的模型类型"""
    config_type = config.get("config_type")
    if config_type == "OpenAI":
        return "openai"
    elif config_type == "AzureOpenAI":
        return "aoai"
    elif config_type == "Groq":
        return "groq"
    elif config_type == "Ollama":
        return "ollama"
    elif config_type == "OpenAI-Like":
        return "openai-like"
    else:
        raise ValueError(f"Invalid config type: {config_type}")
        # 如果 config_type 为 None，尝试根据其他字段推断
        # if "api_type" in config and config["api_type"] == "azure":
        #     return "aoai"
        # elif "base_url" in config and "azure" in config["base_url"]:
        #     return "aoai"
        # elif "base_url" in config and "groq" in config["base_url"]:
        #     return "groq"
        # elif "base_url" in config and "ollama" in config["base_url"]:
        #     return "ollama"
        # else:
        #     return "openai"  # 默认返回 openai


def get_client_config_model(config: Dict) -> LLMConfigType:
    """从配置自动选择模型"""
    model_type = get_client_config(config)
    model_class = SUPPORTED_SOURCES[model_type.lower()]
    
    # 确保配置符合指定类型的要求
    try:
        return model_class.model_validate(config)
    except ValidationError as e:
        raise ValueError(f"Configuration is not valid for {model_type}: {str(e)}")
