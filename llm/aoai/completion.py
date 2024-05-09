import autogen

from typing import List, Dict, Union, Literal
from llm.aoai.Agent.Agent import *
from autogen.oai import OpenAIWrapper
from autogen.oai.openai_utils import config_list_from_dotenv

# def create_completion(prompt: str,
#                       model_type: Literal["OpenAI","Ollama"],
#                       session_state: SessionStateStatProvider):
#     if model_type == "OpenAI":
#         client = OpenAIWrapper(api_key=os.getenv('AZURE_OAI_KEY'),
#                        base_url=os.getenv('AZURE_OAI_ENDPOINT'),
#                        api_version=os.getenv('API_VERSION'),
#                        api_type=os.getenv('API_TYPE'))

def aoai_config_generator(**kwargs):
    '''
    生成符合 Autogen 规范的Groq Completion Client配置

    Args:
        kwargs (dict): 配置参数
            model (str): 模型名称
            api_key (str): API Key
            base_url (str): 模型终结点
            api_type (str): 默认为 "azure"
            api_version (str): API Version, 默认为 "2024-02-15-preview"
        
    Returns:
        config (list): 配置列表
    '''
    config = {
        "model": kwargs.get("model", "gpt-3.5-turbo"),
        "api_key": os.getenv("AZURE_OAI_KEY",default=kwargs.get("api_key","noaoaikey")),
        "base_url": os.getenv("AZURE_OAI_ENDPOINT",default=kwargs.get("base_url","noaoaiendpoint")),
        "api_type": os.getenv("API_TYPE",default=kwargs.get("api_type","azure")),
        "api_version": os.getenv("API_VERSION",default=kwargs.get("api_version","2024-02-15-preview")),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": kwargs.get("stream", False),
        }
    }
    return [config]


class AzureOpenAICompletionClient:
    '''用于生成 Azure OpenAI 聊天补全的基本类'''
    def __init__(self):
        '''使用 Autogen 的 Wrapper 创建相应的CompletionClient'''
        # self.chat_history = session_state["messages"]
        # '''chat_history: 包括`当前轮次对话`的聊天历史'''
        self.client = OpenAIWrapper(api_key=os.getenv('AZURE_OAI_KEY'),
                                    base_url=os.getenv('AZURE_OAI_ENDPOINT'),
                                    api_version=os.getenv('API_VERSION'),
                                    api_type=os.getenv('API_TYPE'))
        '''client: 使用 Autogen 的 OpenAI 包装器的客户端'''
        
    def create_completion(self, 
                          messages: List[Dict[str, str]],
                          model: str):
        '''
        创建补全
        
        Args:
            chat_history (List[Dict[str, str]]): 包括`当前轮次对话`的聊天历史
            model (str): 模型名称
            
        Returns:
            response (dict): 补全的文本
            response_cost (float): 补全的文本的计费成本
        '''
        raw_response = self.client.create(
            model=model,
            # 暂时没有 system prompt
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in messages
            ],
            stream=True
        )
        return raw_response