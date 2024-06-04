from typing import Literal, List, Dict, Any, Generator
from abc import ABC, abstractmethod
import os
import json

from api.dependency import APIRequestHandler,SUPPORTED_SOURCES


class OpenAILikeModelConfigProcessStrategy(ABC):
    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def update_config(self):
        pass

    @abstractmethod
    def delete_model_config(self):
        pass

    @abstractmethod
    def get_model_config(self):
        pass


class ChatProcessStrategy(ABC):
    @abstractmethod
    def create_completion(self, messages: List[Dict[str, str]]) -> Dict:
        pass


class AgentChatProcessoStrategy(ABC):
    @abstractmethod
    def create_reflection_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def create_rag_agent_response(self) -> Dict:
        pass

    @abstractmethod
    def create_base_chat_agent_response(self) -> Dict:
        pass


class ChatProcessor(ChatProcessStrategy):
    """
    处理聊天消息的策略模式实现类
    """
    def __init__(
            self, 
            requesthandler: APIRequestHandler,
            model_type: str,
            llm_config: Dict
        ) -> None:
        self.requesthandler = requesthandler
        self.model_type = model_type
        self.llm_config = llm_config

    def create_completion(
            self, 
            messages: List[Dict[str, str]],
        ) -> Dict:
        """
        通过 requesthandler ，根据 model_type ，调用相应的 API 接口
        """
        if self.model_type.lower() in SUPPORTED_SOURCES["sources"]:

            # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中的对应值为 "sdk" ，则走 OpenAI 的 SDK
            if SUPPORTED_SOURCES["sources"][self.model_type.lower()] == "sdk":
                path = "/chat/openai-like-chat/openai"

            # 否则，走 request 或另行定义的 SDK （如 Groq）
            else:
                # path = "/chat/openai-like-chat/xxxx"
                pass

        response = self.requesthandler.post(
            path,
            data={
                "llm_config": self.llm_config,
                "llm_params": self.llm_config.get(
                    "params",
                    {
                        "temperature": 0.5,
                        "top_p": 0.1,
                        "max_tokens": 4096
                    }
                ),
                "messages": messages
            }
        )
        
        return response
    

    def create_completion_stream_api(
            self, 
            messages: List[Dict[str, str]],
        ) -> str:
        """
        通过 requesthandler ，根据 model_type ，调用相应的 API 接口创建流式输出
        """
        if self.model_type.lower() in SUPPORTED_SOURCES["sources"]:
            # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中的对应值为 "sdk" ，则走 OpenAI 的 SDK
            if SUPPORTED_SOURCES["sources"][self.model_type.lower()] == "sdk":
                path = "/chat/openai-like-chat/openai/stream"

            # 否则，走 request 或另行定义的 SDK （如 Groq）
            else:
                # path = "/chat/openai-like-chat/xxxx/stream"
                pass
            
        response = self.requesthandler.post(
            path,
            data={
                "llm_config": self.llm_config,
                "llm_params": self.llm_config.get(
                    "params",
                    {
                        "temperature": 0.5,
                        "top_p": 0.1,
                        "max_tokens": 4096,
                        "stream": True
                    }
                ),
                "messages": messages
            }
        )
        
        return response
    

    def create_completion_stream_noapi(
            self, 
            messages: List[Dict[str, str]],
        ) -> Generator:
        """
        通过 requesthandler ，根据 model_type ，调用相应的 SDK (不经过后端API)接口创建流式输出
        """
        if self.model_type.lower() in SUPPORTED_SOURCES["sources"]:
            # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中的对应值为 "sdk" ，则走 OpenAI 的 SDK
            if SUPPORTED_SOURCES["sources"][self.model_type.lower()] == "sdk":
                if self.llm_config.get("api_type") != "azure":
                    from openai import OpenAI
                    client = OpenAI(
                        api_key=self.llm_config.get("api_key"),
                        base_url=self.llm_config.get("base_url"),
                    )
                    
                    response = client.chat.completions.create(
                        model=self.llm_config.get("model"),
                        messages=messages,
                        temperature=self.llm_config.get("params", {}).get("temperature", 0.5),
                        top_p=self.llm_config.get("params", {}).get("top_p", 0.1),
                        max_tokens=self.llm_config.get("params", {}).get("max_tokens", 4096),
                        stream=True
                    )
                
                else:
                    from openai import AzureOpenAI
                    client = AzureOpenAI(
                        api_key=self.llm_config.get("api_key"),
                        azure_endpoint=self.llm_config.get("base_url"),
                        api_version=self.llm_config.get("api_version"),
                    )

                    response = client.chat.completions.create(
                        # TODO: Azure 的模型名称是 deployment name ，可能需要自定义
                        model=self.llm_config.get("model"),
                        messages=messages,
                        temperature=self.llm_config.get("params", {}).get("temperature", 0.5),
                        top_p=self.llm_config.get("params", {}).get("top_p", 0.1),
                        max_tokens=self.llm_config.get("params", {}).get("max_tokens", 4096),
                        stream=True
                    )

                return response
        


class AgentChatProcessor(AgentChatProcessoStrategy):
    """
    处理 Agent Chat 消息的策略模式实现类
    """
    def __init__(
        self,
        requesthandler: APIRequestHandler,
        model_type: str,
        llm_config: Dict
    ) -> None:
        self.requesthandler = requesthandler
        self.model_type = model_type
        self.llm_config = llm_config
        
    def create_rag_agent_response(
        self,
        name: str,
        messages: List[Dict[str, str]],
        is_rerank: bool,
        is_hybrid_retrieve: bool,
        hybrid_retriever_weight: float = 0.5,
    ) -> Dict[str, Any]:
        """
        通过 requesthandler ，根据 model_type ，创建一个agentchat的lc-rag响应

        Args:
            name (str): 使用到的 Chroma collection 的名称
            messages (List[Dict[str, str]]): 对话消息列表
            is_rerank (bool): 是否进行重排序
            is_hybrid_retrieve (bool): 是否进行混合检索
            hybrid_retriever_weight (float): 混合检索的权重
        """
        if self.model_type.lower() == "llamafile" or "ollama":
            response = self.requesthandler.post(
                endpoint="/agentchat/lc-rag/create-rag-response",
                data={
                    "messages": messages,
                    "llm_config": self.llm_config,
                    "llm_params": self.llm_config.get(
                        "params",
                        {
                            "temperature": 0.5,
                            "top_p": 0.1,
                            "max_tokens": 4096
                        }
                    )
                },
                params={
                    "name": name,
                    "is_rerank": is_rerank,
                    "is_hybrid_retrieve": is_hybrid_retrieve,
                    "hybrid_retriever_weight": hybrid_retriever_weight,
                }
            )
            return response
    
    def create_function_call_agent_response(
        self,
        message: Dict[str, str] | str,
        tools: List
    ) -> List[Dict[str, str]]:
        '''
        通过 requesthandler ，创建一个agentchat的function call响应
        '''
        response = self.requesthandler.post(
            endpoint="/agentchat/autogen/create-function-call-agent-response",
            data={
                "message": message,
                "llm_config": self.llm_config,
                "llm_params": self.llm_config.get(
                    "params",
                    {
                        "temperature": 0.5,
                        "top_p": 0.1,
                        "max_tokens": 4096
                    }
                ),
                "tools": tools
            }
        )
        return response
    
    def create_function_call_agent_response_noapi(
            self,
            message: str | Dict,
            tools: List,
        ) -> Dict:
        from llm.Agent.pre_built import create_function_call_agent_response
        from utils.basic_utils import config_list_postprocess
        # 对config进行处理
        try:
            config_list = config_list_postprocess([self.llm_config])
        except:
            config_list = [self.llm_config]

        response = create_function_call_agent_response(
            message=message,
            config_list=config_list,
            tools=tools
        )
        return response.chat_history
    
    def create_base_chat_agent_response(self) -> Dict:
        pass

    def create_reflection_agent_response(self) -> Dict:
        pass


class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """
    config_path = "custom_model_config.json"

    def __init__(self):
        # 如果本地没有custom_model_config.json文件，则创建
        if not os.path.exists(self.config_path):
            with open(self.config_path, "w") as f:
                json.dump({}, f)
        
        # 读取custom_model_config.json文件
        with open(self.config_path, "r") as f:
            self.exist_config = json.load(f)
    
    def reinitialize(self) -> None:
        """
        重新初始化类实例
        """
        self.__init__()

    def get_config(self) -> Dict:
        """
        获取完整的配置文件配置信息
        """
        return self.exist_config
    
    def update_config(
            self, 
            model:str,
            base_url: str,
            api_key: str,
        ) -> None:
        """
        更新模型的配置信息
        
        Args:
            config (Dict): 模型的配置信息
                该字典以model为key，以model的配置信息为value:
                {
                    "deepseek-chat": {
                        "base_url": "https://api.deepseek.com/v1/",
                        "api_key": "your_api_key"
                    }
                }
        """
        config = {
            "base_url": base_url,
            "api_key": api_key
        }
        self.exist_config[model] = config
        
        # 更新custom_model_config.json文件
        with open(self.config_path, "w") as f:
            json.dump(self.exist_config, f)
        
    def delete_model_config(self, model:str) -> None:
        """
        删除模型的配置信息
        """
        if model in self.exist_config:
            del self.exist_config[model]
            
            # 更新custom_model_config.json文件
            with open(self.config_path, "w") as f:
                json.dump(self.exist_config, f)
                
    def get_model_config(self,model:str) -> Dict:
        """
        获取指定模型的配置信息
        """
        dict_body = self.exist_config.get(model, {})
        return {model: dict_body}
