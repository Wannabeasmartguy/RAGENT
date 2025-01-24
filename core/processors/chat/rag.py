import os
import json
import copy
from typing import List, Dict, Optional

from core.strategy import RAGChatProcessStrategy
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


class RAGChatProcessor(RAGChatProcessStrategy):
    """
    处理 Agent Chat 消息的策略模式实现类
    """

    def __init__(self, model_type: str, llm_config: Dict) -> None:
        self.model_type = model_type
        self.llm_config = llm_config

    def _parse_llm_config(self, llm_config: Dict) -> Dict:
        """
        解析LLM配置，返回config和params
        """
        config_copy = copy.deepcopy(llm_config)
        params = config_copy.pop("params")
        params.pop("stream")
        return config_copy, params

    def _parse_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        解析messages，返回context_messages和user_prompt
        """
        messages_copy = copy.deepcopy(messages)
        if messages_copy[-1]["role"] == "user":
            user_prompt = messages_copy.pop(-1)["content"]
        else:
            raise ValueError("Last message must be user")
        return messages_copy, user_prompt

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
        """
        使用完全自定义的 RAG 模块，创建一个 RAG 响应
        """
        # 处理messages
        context_messages, user_prompt = self._parse_messages(messages)

        # 处理config
        config_copy, params = self._parse_llm_config(self.llm_config)

        # 创建LLM
        if "api_type" in config_copy:
            if config_copy["api_type"] == "azure":
                llm = AzureOpenAILLM(**config_copy, **params)
        else:
            llm = OpenAILLM(**config_copy, **params)

        # 创建retriever
        # 先读取embedding配置
        embedding_config_file_path = os.path.join(
            "dynamic_configs", "embedding_config.json"
        )
        with open(embedding_config_file_path, "r", encoding="utf-8") as f:
            embedding_config = json.load(f)
        try:
            knowledge_bases = embedding_config.get("knowledge_bases", [])
            collection_config = next(
                (kb for kb in knowledge_bases if kb.get("name") == collection_name),
                None,
            )
            # 实际要传入的collection_name是collection_name的值，而不是collection_name的key
            collection_id = collection_config.get("id")

            if not collection_config:
                raise ValueError(
                    f"在embedding_config中没有找到collection_name: {collection_name} 的配置"
                )

            model_id = collection_config.get("embedding_model_id")
            models = embedding_config.get("models", [])
            embedding_model = next(
                (model for model in models if model.get("id") == model_id), None
            )

            if not embedding_model:
                raise ValueError(
                    f"在embedding_config中没有找到id为 {model_id} 的embedding模型配置"
                )

            embedding_model_or_path = embedding_model.get(
                "embedding_model_name_or_path"
            )
            embedding_type = embedding_model.get("embedding_type")
            if not embedding_model_or_path:
                raise ValueError(
                    f"embedding模型 {model_id} 缺少embedding_model_name_or_path配置"
                )
        except Exception as e:
            raise ValueError(f"处理embedding配置时出错: {str(e)}") from e

        # 如果id在models中对应的embedding_type是sentence_transformer, 则路径需要加上"embeddings/"
        if embedding_model.get("embedding_type") == "sentence_transformer":
            embedding_model_or_path = os.path.join(
                "embeddings", embedding_model_or_path
            )
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
                where={"source": {"$eq": selected_file}}
            )
        if is_hybrid_retrieve:
            bm25_retriever = BM25Retriever.from_texts(
                texts=retriever.retriever.collection.get()["documents"],
                metadatas=retriever.retriever.collection.get()["metadatas"],
            )
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=[hybrid_retriever_weight, 1 - hybrid_retriever_weight],
            )
        if is_rerank:
            reranker = BgeRerank()
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=retriever
            )

        # 创建RAG
        rag_builder = RAGBuilder()
        rag = (
            rag_builder.with_llm(llm)
            .with_retriever(retriever)
            .for_rag_type("ConversationRAG")
            .build()
        )

        response = rag.invoke(query=user_prompt, stream=stream)
        return response
