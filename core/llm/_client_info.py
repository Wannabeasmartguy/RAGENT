from typing import Literal
from core.model.llm import *

SUPPORTED_SOURCES = {
    "openai": OpenAIConfig,
    "aoai": AzureOpenAIConfig,
    "llamafile": OpenAIConfig,
    "ollama": OllamaConfig,
    "groq": GroqConfig,
    "openai-like": OpenAIConfig
}

def generate_client_config(
        source: Literal["openai", "aoai", "llamafile", "ollama", "groq", "openai-like"], 
        **kwargs
    ) -> LLMBaseConfig:
    '''
    生成客户端配置

    Args:
        source (str): 数据源
        **kwargs: 配置参数
    Returns:
        config (LLMBaseConfig): LLM配置
    '''
    return SUPPORTED_SOURCES[source].from_env(**kwargs)
