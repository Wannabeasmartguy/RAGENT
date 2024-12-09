import os

def oai_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的OpenAI Completion Client配置

    Args:
        kwargs (dict): 配置参数
            model (str): 模型名称
            api_key (str): API Key
            temperature (float): 温度
            top_p (float): top_p
            stream (bool): 是否流式输出
        
    Returns:
        config (list): 配置列表
    '''
    config = {
        "model": kwargs.get("model", "gpt-3.5-turbo"),
        "api_key": os.getenv("OPENAI_API_KEY",default=kwargs.get("api_key","noaoaikey")),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 1.0),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": kwargs.get("stream", False),
        }
    }
    return [config]