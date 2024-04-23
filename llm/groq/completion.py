import os

from groq import Groq
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()

def groq_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的Groq Completion Client配置

    Args:
        kwargs (dict): 配置参数
            model (str): 模型名称
            api_key (str): API Key
            temperature (float): 温度
            top_p (float): Top P
            stream (bool): 是否流式输出
        
    Returns:
        config (list): 配置列表
    '''
    config = {
        "model": kwargs.get("model", "llama3-8b-8192"),
        "api_key": os.getenv("GROQ_API_KEY",default=kwargs.get("api_key","nogroqkey")),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        },
        "model_client_cls": "GroqClient",
    }
    return [config]

class GroqClient:
    '''符合 Autogen 规范的Groq Completion Client.'''
    def __init__(self,config: dict):
        # print(f"GroqCompletionClient config: {config}")
        self.model = config['model']
        self.client = Groq(api_key=config['api_key'])

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
    
class GroqCompletionClient(GroqClient):
    '''专门用于Groq的Completion Client'''
    def __init__(self,config):
        super().__init__(config)

    def create_completion(self, **config):
        # 先获得父类的 create 输出
        output = super().create(config)
        # 然后从输出中提取消息，放进 Namespace
        response = output.choices[0].message.content
        cost = 0
        return SimpleNamespace(response=response, cost=cost)
    
    def extract_text_or_completion_object(self, response:SimpleNamespace):
        '''
        从 SimpleNamespace 对象中提取文本或 Completion 对象
        '''
        return [response.response]