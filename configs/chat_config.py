from typing import Literal, List, Dict, Any, Generator, Optional
from functools import partial
from uuid import uuid4
import os
import json
import requests
import copy

from autogen import OpenAIWrapper
from groq import Groq

from api.dependency import APIRequestHandler, SUPPORTED_SOURCES
from api.routers.chat import LLMConfig, LLMParams
from configs.strategy import (
    ChatProcessStrategy,
    AgentChatProcessoStrategy,
    OpenAILikeModelConfigProcessStrategy,
    CozeChatProcessStrategy
)
from configs.basic_config import (
    CONFIGS_BASE_DIR,
    CONFIGS_DB_FILE,
    EMBEDDING_CONFIGS_DB_TABLE,
    LLM_CONFIGS_DB_TABLE
)
from storage.db.sqlite import (
    SqlAssistantLLMConfigStorage,
    SqlEmbeddingConfigStorage
)
from model.config.llm import OpenAILikeLLMConfiguration
from utils.tool_utils import create_tools_call_completion
from tools.toolkits import TOOLS_LIST, TOOLS_MAP

from modules.rag.builder.builder import RAGBuilder
from modules.llm.openai import OpenAILLM
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
            endpoint=path,
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
    
    def create_completion_noapi(
        self,
        messages: List[dict]
    ):
        '''
        创建一个 LLM 模型，并使用该模型生成一个响应。
        
        Args:
            source (str):  支持的 LLM 推理源，保存于 dependence.py 中。
            llm_config (LLMConfig): LLM 模型的配置信息。
            llm_params (LLMParams, optional): LLM 模型的参数信息，包括 temperature、top_p 和 max_tokens。
            messages (List[dict]): 完整的对话消息列表。
            support_sources (dict): 支持的 LLM 源列表。
            
        Returns:
            Dict: 生成的响应。
        '''
        llm_config = LLMConfig(**self.llm_config)
        llm_params = LLMParams(**self.llm_config.get("params", {}))
        
        # 检查 source 是否在支持列表中
        source = self.model_type.lower()

        if source not in SUPPORTED_SOURCES["sources"]:
            raise ValueError(f"Unsupported source: {source}")
        
        # 根据 source 选择不同的处理逻辑
        # 如果 Source 在 sources 中为 "sdk"，则使用 OpenAI SDK 进行处理
        # 如果 Source 在 sources 中为 "request"，则使用 Request 进行处理

        if SUPPORTED_SOURCES["sources"][source] == "sdk":
            client = OpenAIWrapper(
                **llm_config.dict(exclude_unset=True),
                # 禁用缓存
                cache = None,
                cache_seed = None
            )

            if llm_params:
                response = client.create(
                    messages = messages,
                    model = llm_config.model,
                    temperature = llm_params.temperature,
                    top_p = llm_params.top_p,
                    max_tokens = llm_params.max_tokens
                )
            else:
                response = client.create(
                    messages = messages,
                    model = llm_config.model,
                )

            return response
        
        # elif SUPPORTED_SOURCES["sources"][source] == "groq":
        #     client = Groq(
        #         api_key=llm_config.api_key,
        #     )

        #     if llm_params:
        #         response = client.chat.completions.create(
        #             messages = messages,
        #             model = llm_config.model,
        #             temperature = llm_params.temperature,
        #             top_p = llm_params.top_p,
        #             max_tokens = llm_params.max_tokens
        #         )
        #     else:
        #         response = client.chat.completions.create(
        #             messages = messages,
        #             model = llm_config.model,
        #         )
            
        #     return response

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
            endpoint=path,
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
                        model=self.llm_config.get("model").replace(".", ""),
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
    
    def create_custom_rag_response(
        self,
        *,
        collection_name: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        is_rerank: bool = False,
        is_hybrid_retrieve: bool = False,
        hybrid_retriever_weight: float = 0.5,
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
            collection_config = embedding_config.get(collection_name, {})
            embedding_model_or_path = collection_config.get("embedding_model_name_or_path")
        except Exception as e:
            raise ValueError(f"embedding_config_file_path: {embedding_config_file_path} 中没有找到collection_name: {collection_name} 的配置") from e
        
        retriever = ChromaContextualRetriever(
            llm=llm,
            collection_name=collection_name,
            embedding_model=embedding_model_or_path,
        )
        if is_hybrid_retrieve:
            bm25_retriever = BM25Retriever.from_texts(
                texts=retriever.retriever.collection.get()['documents'],
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
        retriever.update_context_messages(context_messages)

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


class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """
    config_path = os.path.join("dynamic_configs", "custom_model_config.json")

    def __init__(self):
        # 如果本地没有custom_model_config.json文件，则创建文件夹及文件
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
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


class OAILikeSqliteConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类，数据保存在 SQlite 中
    """
    def __init__(
            self,
            table_name: str = LLM_CONFIGS_DB_TABLE,
            db_url: Optional[str] = None,
            db_file: Optional[str] = CONFIGS_DB_FILE
        ):
        self.storage = SqlAssistantLLMConfigStorage(
            table_name=table_name,
            db_url=db_url,
            db_file=db_file
        )
        self.models_table = self.storage.get_all_models()
    
    def reinitialize(self) -> None:
        """
        重新初始化类实例
        """
        self.__init__()

    def get_config(self) -> Dict:
        """
        获取完整的配置文件配置信息的字典
        """
        model_config = {}
        for model in self.models_table:
            model_info = model.dict()
            model_config[model.model_name] = model_info
        return model_config
    
    def update_config(
        self,
        model_name:str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        更新模型的配置信息
        
        Args:
            model_name (str): 模型的名称
            api_key (str): 模型的API Key
            base_url (Optional[str], optional): 访问模型的端点.
            kwargs (Dict): 其他配置信息
        """
        oai_like_config = OpenAILikeLLMConfiguration(
            model_id=str(uuid4()),
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self.storage.upsert(oai_like_config)

        

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