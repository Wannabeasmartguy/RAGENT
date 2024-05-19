from openai import OpenAI
from types import SimpleNamespace


def llamafile_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的llamafile Completion Client配置

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
        "base_url": kwargs.get("base_url","http://127.0.0.1:8080/v1"),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        },
        "model_client_cls": "LlamafileClient",
    }
    return [config]


class LlamafileClient:
    '''符合 Autogen 规范的 llamafile Completion Client .'''
    def __init__(self,config: dict):
        self.model = config.get("model","noneed")
        self.client = OpenAI(
            base_url=config.get("base_url","http://127.0.0.1:8080/v1"),
            api_key=config.get("api_key","noneed")
        )

        get_config_param:dict = config.get("params",{})
        self.temperature = get_config_param.get("temperature",0.5)
        # self.max_tokens = get_config_param.get("max_tokens",4000)
        self.top_p = get_config_param.get("top_p",1.0)
        self.stream = get_config_param.get("stream",False)

    def create(self,config:dict) -> dict:
        '''
        创建一个会话
        
        Args:
            config (dict): 配置参数，必须要包含 'messages' 键，其值为一个包含对话消息的列表
            
        Returns:
            dict: 包含会话结果的响应
        '''
        response = self.client.chat.completions.create(
            model=self.model,
            messages=config["messages"],
            temperature=config.get("temperature",self.temperature),
            top_p=config.get("top_p",self.top_p),
            stream=config.get("stream",self.stream),
        )
        return response
    
    def message_retrieval(self,response):
        '''从响应中提取消息'''
        choices = response.choices
        return [choice.message.content for choice in choices]
    
    def cost(self,response) -> float:
        '''
        计算成本(目前Groq不收费)        
        '''
        response.cost = 0
        return response.cost
    
    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}
    
class LlamafileCompletionClient(LlamafileClient):
    '''专门用于 llamafile 的 Completion Client. '''
    def __init__(self,config):
        super().__init__(config)

    def create_completion(self, **config):
        # 先获得父类的 create 输出
        output = super().create(config)
        # 然后从输出中提取消息，放进 Namespace
        response = output.choices[0].message.content
        cost = 0
        return SimpleNamespace(response=response, cost=cost)
    
    def create_completion_stream(self, **config):
        # 先获得父类的 create 输出
        output = super().create(config)
        # 然后从输出中提取消息，放进 Namespace
        if config.get("stream",False):
            cost = 0
            response_text = ""
            for chunk in output:
                if chunk is not None:
                    response = chunk.choices[0].delta.content
                    # yield SimpleNamespace(response=response, cost=cost)
                    if response is not None:
                        response_text += response
                        yield response_text
    
    def extract_text_or_completion_object(self, response:SimpleNamespace):
        '''
        从 SimpleNamespace 对象中提取文本或 Completion 对象
        '''
        return [response.response]