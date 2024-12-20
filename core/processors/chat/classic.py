from typing import Literal, List, Dict, Any, Generator, Optional
from functools import partial
from uuid import uuid4
from copy import deepcopy
from deprecated import deprecated
import os
import json
import uuid
import requests
import copy

from loguru import logger

from api.dependency import APIRequestHandler
from core.llm._client_info import SUPPORTED_SOURCES as SUPPORTED_CLIENTS
from core.llm._client_info import OPENAI_SUPPORTED_CLIENTS
from core.strategy import (
    ChatProcessStrategy,
    AgentChatProcessStrategy,
    OpenAILikeModelConfigProcessStrategy,
    CozeChatProcessStrategy
)
from core.strategy import EncryptorStrategy
from core.encryption import FernetEncryptor
from utils.tool_utils import create_tools_call_completion
from utils.log.logger_config import setup_logger

from modules.rag.builder.builder import RAGBuilder
from modules.llm.openai import OpenAILLM
from modules.llm.aoai import AzureOpenAILLM
from modules.retrievers.vector.chroma import ChromaRetriever, ChromaContextualRetriever
from modules.retrievers.bm25 import BM25Retriever
from modules.rerank.bge import BgeRerank
from modules.retrievers.emsemble import EnsembleRetriever
from modules.retrievers.comtextual_compression import ContextualCompressionRetriever
from modules.rag.builder.builder import RAGBuilder
from modules.types.rag import BaseRAGResponse


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
        self.create_tools_call_completion = partial(create_tools_call_completion, config_list=[llm_config])
    
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict | Generator:
        '''
        创建一个 LLM 响应，支持普通输出和流式输出。
        
        Args:
            messages (List[Dict[str, str]]): 完整的对话消息列表
            stream (bool, optional): 是否使用流式输出。默认为 False
            
        Returns:
            Dict | Generator: 根据 stream 参数返回不同类型的响应
        '''
        source = self.model_type.lower()
        if source not in SUPPORTED_CLIENTS:
            raise ValueError(f"Unsupported source: {source}")
        
        if source in OPENAI_SUPPORTED_CLIENTS:
            try:
                if self.llm_config.get("api_type") != "azure":
                    from openai import OpenAI
                    client = OpenAI(
                        api_key=self.llm_config.get("api_key"),
                        base_url=self.llm_config.get("base_url"),
                    )
                else:
                    from openai import AzureOpenAI
                    client = AzureOpenAI(
                        api_key=self.llm_config.get("api_key"),
                        azure_endpoint=self.llm_config.get("base_url"),
                        api_version=self.llm_config.get("api_version"),
                    )
                    
                params = {
                    "model": self.llm_config.get("model").replace(".", "") if self.llm_config.get("api_type") == "azure" else self.llm_config.get("model"),
                    "messages": messages,
                    "temperature": self.llm_config.get("params", {}).get("temperature", 0.5),
                    "top_p": self.llm_config.get("params", {}).get("top_p", 0.1),
                    "max_tokens": self.llm_config.get("params", {}).get("max_tokens", 4096),
                    "stream": stream
                }
                
                response = client.chat.completions.create(**params)
                return response
                
            except Exception as e:
                raise ValueError(f"Error creating completion: {str(e)}") from e


@deprecated("AgentChatProcessor is deprecated. Use AgentChatProcessor in core.processors.chat.agent instead.")
class AgentChatProcessor(AgentChatProcessStrategy):
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
    
    def create_custom_rag_response(
        self,
        *,
        collection_name: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        is_rerank: bool = False,
        is_hybrid_retrieve: bool = False,
        hybrid_retriever_weight: float = 0.5,
        selected_file: Optional[str] = None,
    ) -> BaseRAGResponse:
        '''
        使用完全自定义的 RAG 模块，创建一个 RAG 响应
        '''
        # 处理messages
        messages_copy = copy.deepcopy(messages)
        user_prompt = messages_copy.pop(-1)["content"]
        context_messages = messages_copy

        # 处理config
        config_copy = copy.deepcopy(self.llm_config)
        params = config_copy.pop("params")
        params.pop("stream")

        # 创建LLM
        if "api_type" in config_copy:
            if config_copy["api_type"] == "azure":
                llm = AzureOpenAILLM(
                    **config_copy,
                    **params
                )
        else:
            llm = OpenAILLM(
                **config_copy,
                **params
            )

        # 创建retriever
        # 先读取embedding配置
        embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")
        with open(embedding_config_file_path, "r", encoding="utf-8") as f:
            embedding_config = json.load(f)
        try:
            knowledge_bases = embedding_config.get("knowledge_bases", [])
            collection_config = next((kb for kb in knowledge_bases if kb.get("name") == collection_name), None)
            # 实际要传入的collection_name是collection_name的���，而不是collection_name的key
            collection_id = collection_config.get("id")

            if not collection_config:
                raise ValueError(f"在embedding_config中没有找到collection_name: {collection_name} 的配置")
            
            model_id = collection_config.get("embedding_model_id")
            models = embedding_config.get("models", [])
            embedding_model = next((model for model in models if model.get("id") == model_id), None)
            
            if not embedding_model:
                raise ValueError(f"在embedding_config中没有找到id为 {model_id} 的embedding模型配置")
            
            embedding_model_or_path = embedding_model.get("embedding_model_name_or_path")
            embedding_type = embedding_model.get("embedding_type")
            if not embedding_model_or_path:
                raise ValueError(f"embedding模型 {model_id} 缺少embedding_model_name_or_path配置")
        except Exception as e:
            raise ValueError(f"处理embedding配置时出错: {str(e)}") from e
        
        # 如果id在models中对应的embedding_type是sentence_transformer, 则路径需要加上"embeddings/"
        if embedding_model.get("embedding_type") == "sentence_transformer":
            embedding_model_or_path = os.path.join("embeddings", embedding_model_or_path)
            embedding_type = "sentence_transformer"
        
        retriever = ChromaContextualRetriever(
            llm=llm,
            collection_name=collection_id,
            embedding_model=embedding_model_or_path,
            embedding_type=embedding_type,
        )
        retriever.update_context_messages(context_messages)
        
        if selected_file:
            retriever.retriever.update_parameters(
                where={
                    "source": {
                        "$eq": selected_file
                    }
                }
            )
        if is_hybrid_retrieve:
            bm25_retriever = BM25Retriever.from_texts(
                texts=retriever.retriever.collection.get()['documents'],
                metadatas=retriever.retriever.collection.get()['metadatas'],
            )
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=[hybrid_retriever_weight, 1 - hybrid_retriever_weight],
            )
        if is_rerank:
            reranker = BgeRerank()
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever
            )

        # 创建RAG
        rag_builder = RAGBuilder()
        rag = rag_builder.with_llm(llm).with_retriever(retriever).for_rag_type("ConversationRAG").build()

        response = rag.invoke(query=user_prompt, stream=stream)
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


@deprecated("OAILikeConfigProcessor is deprecated. Use OAILikeConfigProcessor in core.processors.config.llm instead.")
class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """
    config_path = os.path.join("dynamic_configs", "custom_model_config.json")

    def __init__(self, encryptor: EncryptorStrategy = None):
        """
        Args:
            encryptor (EncryptorStrategy, optional): Defaults to None. If not provided, a new FernetEncryptor will be created.
        """
        self.encryptor = encryptor or FernetEncryptor()
        # 如果本地没有custom_model_config.json文件，则创建文件夹及文件
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump({}, f, indent=4)
        
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
            model: str,
            base_url: str,
            api_key: str,
            custom_name: str = "",
            description: str = "",
        ) -> str:
        """
        更新或添加模型的配置信息
        
        Args:
            model (str): 模型名称
            base_url (str): API基础URL
            api_key (str): API密钥
            description (str): 配置描述，用于区分相同模型的不同配置

        Returns:
            str: 配置的唯一标识符
        """
        config_id = str(uuid.uuid4())
        config = {
            "model": model,
            "base_url": self.encryptor.encrypt(base_url),
            "api_key": self.encryptor.encrypt(api_key),
            "custom_name": custom_name,
            "description": description
        }
        self.exist_config[config_id] = config
        
        # 更新custom_model_config.json文件
        with open(self.config_path, "w") as f:
            json.dump(self.exist_config, f, indent=4)
        
        return config_id
        
    def delete_model_config(self, config_id: str) -> None:
        """
        删除模型的配置信息
        """
        if config_id in self.exist_config:
            del self.exist_config[config_id]
            
            # 更新custom_model_config.json文件
            with open(self.config_path, "w") as f:
                json.dump(self.exist_config, f, indent=4)
                
    def get_model_config(self, model: str = None, config_id: str = None) -> Dict:
        """
        获取指定模型或配置ID的配置信息
        
        Args:
            model (str, optional): 模型名称
            config_id (str, optional): 配置ID

        Returns:
            Dict: 匹配的配置信息字典
        """
        if config_id:
            config = self.exist_config.get(config_id, {})
            if config:
                config["base_url"] = self.encryptor.decrypt(config["base_url"])
                config["api_key"] = self.encryptor.decrypt(config["api_key"])
            return config
        elif model:
            return {
                config_id: {
                    **config,
                    "base_url": self.encryptor.decrypt(config["base_url"]),
                    "api_key": self.encryptor.decrypt(config["api_key"]),
                }
                for config_id, config in self.exist_config.items()
                if config["model"] == model
            }
        else:
            return {}

    def list_model_configs(self) -> List[Dict]:
        """
        列出所有模型配置
        
        Returns:
            List[Dict]: 包含所有配置信息的列表
        """
        return [
            {
                "id": config_id, 
                **{k: v if k not in ['base_url', 'api_key'] else '******' for k, v in config.items()}
            }
            for config_id, config in self.exist_config.items()
        ]
        

class CozeChatProcessor(CozeChatProcessStrategy):
    """
    处理与 Coze API 相关的逻辑
    """
    def __init__(
            self, 
            access_token: Optional[str] = None,
        ):
        
        if access_token:
            self.personal_access_token = access_token
        else:
            from dotenv import load_dotenv
            load_dotenv()
            self.personal_access_token = os.getenv("COZE_ACCESS_TOKEN")
    
    def create_coze_agent_response(
            self,
            user: str,
            query: str,
            bot_id: str,
            stream: bool = False,
            conversation_id: Optional[str] = None,
            chat_history: Optional[List[Dict[str, str]]] = None,
            custom_variables: Optional[Dict[str, str]] = None,
        ) -> requests.Response:
        """
        Creates a response from the Coze Agent API.

        Args:
            user (str): The user identifier.
            query (str): The user's input or question.
            bot_id (str): Identifier for the bot to use.
            stream (bool, optional): Whether to return streaming responses. Defaults to False.
            conversation_id (Optional[str], optional): ID of the current conversation. Defaults to None.
            chat_history (Optional[List[Dict[str, str]]], optional): History of previous interactions. Defaults to None.
            custom_variables (Optional[Dict[str, str]], optional): Custom variables for the request. Defaults to None.

        Returns:
            requests.Response: The response from the API call.
        """
        headers = {
            'Authorization': f'Bearer {self.personal_access_token}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'api.coze.cn',
            'Connection': 'keep-alive'
        }

        data = {
            'user': user,
            'query': query,
            'bot_id': bot_id,
            'stream': stream,
            'conversation_id': conversation_id,
            'chat_history': chat_history,
            'custom_variables': custom_variables
        }

        data = dict(filter(lambda item: item[1] is not None, data.items()))

        response = requests.post(
            url='https://api.coze.cn/open_api/v2/chat',
            headers=headers,
            json=data
        )

        return response

    @classmethod
    def get_bot_config(
            cls,
            personal_access_token: str,
            bot_id: str,
            bot_version: Optional[str] = None
        ) -> requests.Response:
        headers = {
            'Authorization': f'Bearer {personal_access_token}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        
        params = {
            'bot_id': bot_id,
            'bot_version': bot_version
        }
        params = dict(filter(lambda item: item[1] is not None, params.items()))

        response = requests.get(
            url="https://api.coze.cn/v1/bot/get_online_info",
            headers=headers,
            params=params
        )
        return response