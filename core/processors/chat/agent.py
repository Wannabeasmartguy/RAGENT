from typing import Literal, List, Dict, Any, Generator, Optional
from uuid import uuid4
from core.strategy import AgentChatProcessStrategy
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
from api.dependency import APIRequestHandler
import os
import json
import copy


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
            # 实际要传入的collection_name是collection_name的值，而不是collection_name的key
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
        from core.llm.Agent.pre_built import create_function_call_agent_response
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