from typing import Literal, Dict, Type, List
from typing import Literal, Dict, Type, List
from enum import Enum
import os
import os

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


def parse_env_list(env_key: str, prefix: str = "") -> List[str]:
    """
    解析环境变量中的列表配置
    例如: AZURE_OAI_KEY_LIST="key1,key2,key3"
    """
    value = os.getenv(f"{prefix}{env_key}_LIST", "")
    if not value:
        # 如果没有配置列表，则尝试获取单个值
        single_value = os.getenv(f"{prefix}{env_key}")
        return [single_value] if single_value else []
    return [v.strip() for v in value.split(",") if v.strip()]

def generate_multi_client_configs(
    source: Literal["openai", "azureopenai", "ollama", "groq", "openai-like"],
    **kwargs,
) -> List[LLMBaseConfig]:
    """
    生成多个客户端配置

    Args:
        source (str): 数据源
        **kwargs: 配置参数
    Returns:
        List[LLMBaseConfig]: LLM配置列表
    """
    config_class = SUPPORTED_SOURCES[source]
    
    if source == OpenAISupportedClients.AOAI.value:
        api_keys = parse_env_list("AZURE_OAI_KEY")
        endpoints = parse_env_list("AZURE_OAI_ENDPOINT")
        
        # 确保 keys 和 endpoints 数量匹配
        if len(endpoints) == 1:
            endpoints = endpoints * len(api_keys)
        elif len(api_keys) != len(endpoints):
            raise ValueError("Number of API keys and endpoints must match")
            
        return [
            config_class.from_env(
                api_key=key,
                base_url=endpoint,
                **kwargs
            )
            for key, endpoint in zip(api_keys, endpoints)
        ]
        
    elif source == OpenAISupportedClients.OPENAI.value:
        api_keys = parse_env_list("OPENAI_API_KEY")
        return [
            config_class.from_env(
                api_key=key,
                **kwargs
            )
            for key in api_keys
        ]
        
    elif source == OpenAISupportedClients.GROQ.value:
        api_keys = parse_env_list("GROQ_API_KEY")
        return [
            config_class.from_env(
                api_key=key,
                **kwargs
            )
            for key in api_keys
        ]
    
    # 其他类型的配置可以类似处理...
    # 未作处理的类型则直接使用传入的参数生成单个配置
    return [config_class.from_env(**kwargs)]


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
