from typing import Literal, Dict
from core.model.llm import *

from pydantic import ValidationError


SUPPORTED_SOURCES = {
    "openai": OpenAIConfig,
    "aoai": AzureOpenAIConfig,
    "llamafile": OpenAIConfig,
    "ollama": OllamaConfig,
    "groq": GroqConfig,
    "openai-like": OpenAIConfig,
}

OPENAI_SUPPORTED_CLIENTS = [
    "openai",
    "aoai",
    "llamafile",
    "ollama",
    "groq",
    "openai-like",
]


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
    for source, model_class in SUPPORTED_SOURCES.items():
        try:
            model_class.model_validate(config)
            return source.lower()
        except:
            continue
    raise ValueError("Invalid configuration")
