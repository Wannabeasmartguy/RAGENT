import requests
import json
import streamlit as st
from openai import OpenAI

from typing import List, Dict, Union, Optional


def ollama_config_generator(**kwargs):
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
        "base_url": kwargs.get("base_url","http://localhost:11434/v1"),
        "params": {
            "temperature": kwargs.get("temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.5),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": kwargs.get("stream", False),
        },
        "model_client_cls": "OllamaClient",
    }
    return [config]


def process_chat_response(response):
    """
    处理聊天API的流式输出

    Args:
        response (requests.Response): 聊天API的响应对象

    Yields:
        dict: 每个聊天消息的结果
    """
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data


def process_api_params(is_enable:bool=False,
                       choose_list:list=["max_tokens","frequency_penalty"],
                       **kwargs):
    '''
    处理 Ollama API 传入参数
     OpenAI 的参数和 Ollama 的参数在设置时，合适的数值大小不同
    此外，某些参数的名称也不同，详见 https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    Args:
        is_enable (bool): 是否启用参数处理
        choose_list (list): 需要保留到 API 请求的参数列表
        kwargs (dict): 传入的参数
    '''
    if is_enable:
        # 对option_kwargs进行处理，使其符合API的要求
        num_ctx = kwargs.pop('max_tokens', None)
        if num_ctx is not None:
            kwargs['num_ctx'] = num_ctx
        
        repeat_penalty = kwargs.pop('frequency_penalty', None)
        if repeat_penalty is not None:
            kwargs['repeat_penalty'] = repeat_penalty

        kwargs = {k: v for k, v in kwargs.items() if k in choose_list}

        return kwargs
    

def get_ollama_model_list(url:str="http://localhost:11434/api/"):
    """
    获取模型标签列表
    
    Returns:
        model_list(list): 所有模型的名称
    """
    model_tags_url = f"{url}tags"
    response = requests.get(model_tags_url)

    if response.status_code != 200:
        raise ValueError("无法获取模型标签列表")
    
    if response == None:
        response = {}

    tags = response.json()
    
    # 获取所有模型的名称
    model_list = [model['name'] for model in tags['models']]
    return model_list


class OllamaResponse():
    '''
    用于处理 Ollama API 返回结果的类
    '''
    def __init__(self,response:dict):
        self.response:dict = response
        self.cost:float = 0


class OllamaClient:
    '''符合 Autogen 规范的 llamafile Completion Client .'''
    def __init__(self,config: dict):
        self.model = config.get("model","noneed")
        self.client = OpenAI(
            base_url=config.get("base_url","http://localhost:11434/v1"),
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
        计算成本       
        '''
        response.cost = 0
        return response.cost
    
    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}


class OllamaCompletionClient:
    '''用于生成 Ollama 聊天补全的基本类'''
    def __init__(self):
        self.url:str = "http://localhost:11434/api/"
        self.cost:float = 0

    def create_completion(self,
            messages:list,
            model:str,
            **option_kwargs
        ):
        """
        发送聊天请求

        Args:
            messages (list): 聊天消息列表，每个消息包括角色和内容,与 OpenAI message 格式相同

        Yields:
            dict: 服务器返回的聊天结果
        """
        # url:str="http://localhost:11434/api/chat"
        chat_url = f"{self.url}chat"

        kwargs = process_api_params(is_enable=True, **option_kwargs)

        data = {
            "model": model,
            "messages": messages,
            "stream": False, # 流式输出则注释掉此行
            "options": kwargs
        }

        try:
            response = requests.post(chat_url, json=data, stream=True, timeout=120)
            response_ollama = OllamaResponse(response.json())
        except:
            response = {
                "model": model,
                "message": {
                    "role": "assistant",
                    "content": ""
                }
            }
            response_ollama = OllamaResponse(response)
            st.error("请求超时")

        return response_ollama


    def get_ollama_model_list(self):
        """
        获取模型标签列表
        
        Returns:
            model_list(list): 所有模型的名称
        """
        model_tags_url = f"{self.url}tags"
        response = requests.get(model_tags_url)

        if response.status_code != 200:
            raise ValueError("无法获取模型标签列表")
        
        if response == None:
            response = {}

        tags = response.json()

        # 获取所有模型的名称
        model_list = [model['name'] for model in tags['models']]
        return model_list
    
    def extract_text_or_completion_object(self,response:Union[dict,OllamaResponse]):
        """
        从服务器返回的聊天结果中提取文本或 OllamaResponse 对象
        
        Args:
            response (dict,OllamaResponse): 服务器返回的聊天结果
            
        Returns:
            response_list: 聊天结果的文本列表，一般只有一个消息有文本
        """
        if isinstance(response,dict):
            response_list = [response['message']['content']]
        elif isinstance(response,OllamaResponse):
            response_list = [response.response['message']['content']]
        else:
            raise ValueError("response 参数必须是 dict 或 OllamaResponse 类型")
        return response_list

