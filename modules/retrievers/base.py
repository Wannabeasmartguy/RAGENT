import asyncio
from loguru import logger
from typing import List, Dict, Any, Optional, Coroutine
from modules.llm.openai import OpenAILLM
from collections import defaultdict
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Base class for all retrievers"""
    def __init__(self, docs: List[str]):
        self.docs = [{'page_content': doc} for doc in docs]

    @abstractmethod
    def invoke(self, query: str) -> List[Dict[str, Any]]:
        return self.docs

    @abstractmethod
    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        return self.docs


class BaseContextualRetriever(BaseRetriever):
    """综合聊天记录与用户的最新问题，重写query,使用新query进行检索"""
    def __init__(self, llm: OpenAILLM, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    
    def invoke(
            self, 
            query: str, 
            messages: List[Dict[str, Any]],
            n_results: int = 6,
            where: Optional[Dict] = None,
            where_document: Optional[Dict] = None
        ) -> Dict[str, Any]:
        """重写query,使用新query进行检索"""
        new_query = self._build_contextual_query(query, messages)
        logger.info(f"New query: {new_query}")
        return self.retriever.invoke(
            query_texts=[new_query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )

    def _build_contextual_query(
            self, 
            query: str,
            messages: List[Dict[str, Any]],
            use_llm: bool = True
        ) -> str:
        """构建新的query"""
        # 将messages转换为字符串
        messages_str = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        # 构建新的query
        new_query = (
            "<chat_history>\n"
            f"{messages_str}"
            "\n\n"
            "<query>\n"
            f"{query}\n"
        )
        if use_llm:
            rewritten_query = self.llm.invoke(
                messages=[
                    {"role": "system", "content": self.contextualize_q_system_prompt},
                    {"role": "user", "content": new_query}
                ]
            )
            return rewritten_query.choices[0].message.content
        return new_query
    
    def ainvoke(self, query: str) -> Coroutine[Any, Any, List[Dict[str, Any]]]:
        return super().ainvoke(query)
    
    def invoke_format_to_str(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        n_results: int = 6,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> str:
        """重写query,使用新query进行检索，返回格式化后的字符串"""
        results = self.invoke(
            query=query, 
            messages=messages,
            n_results=n_results, 
            where=where, 
            where_document=where_document
        )
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        return "\n\n".join([f"Document {index+1}: \n{result}" for index, result in enumerate(results['documents'][0])])