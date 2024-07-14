def litellm_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的 litellm Completion Client 配置

    Args:
        kwargs (dict): 配置参数
            model (str): 模型名称
            api_key (str): API Key
            base_url (str): Base URL
            params (dict): 其他请求参数
                temperature (float): 温度
                top_p (float): Top P
                stream (bool): 是否流式输出
        
    Returns:
        config (list): 配置列表
    '''
    config = {
        "model": kwargs.get("model", "noneed"),
        "api_key": kwargs.get("api_key", "noneed"),
        "base_url": kwargs.get("base_url","http://127.0.0.1:4000"),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.5),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": kwargs.get("stream", False),
        }
    }
    return [config]