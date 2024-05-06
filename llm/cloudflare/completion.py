from types import SimpleNamespace
import requests


ACCOUNT_ID = "ACCOUNT_ID here"
AUTH_TOKEN = "AUTH_TOKEN here"

def WorkersAIRequest(prompt:str) -> dict:
    '''
    发起请求
    '''
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3-8b-instruct",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a friendly assistant"},
                {"role": "user", "content": prompt}
            ]
        }
    )
    result = response.json()
    return result


class WorkersAIClient:
    '''符合 Autogen 规范的Groq Completion Client.'''
    def __init__(self,config: dict):
        # print(f"GroqCompletionClient config: {config}")
        self.model = config['model']
        # self.client = Groq(api_key=config['api_key'])

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
        response = WorkersAIRequest(prompt=config['messages'])
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