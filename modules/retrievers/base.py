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
        """
        Invoke the retriever with a query and return the results
        Results format: Dict[str, Any], include two keys: 'page_content' and 'matadata'
        """
        return self.docs

    def _invoke(self, query: str):
        pass

    @abstractmethod
    def invoke_format_to_str(self, query: str) -> Dict[str, Any]:
        """
        Invoke the retriever with a query and return the results in `dict` format, which include a 
        key 'result' is the string of the results.
        Besides, the results also contain key 'page_content' and 'matadata'.
        """
        pass

    @abstractmethod
    async def ainvoke(self, query: str) -> List[Dict[str, Any]]:
        return self.docs


class BaseContextualRetriever(BaseRetriever):
    """综合聊天记录与用户的最新问题，重写query,使用新query进行检索"""
    context_messages = None
    """不包括最新问题的聊天记录"""
    def __init__(
            self, 
            llm: OpenAILLM, 
            retriever: BaseRetriever,
            n_results: int = 6,
            where: Optional[Dict] = None,
            where_document: Optional[Dict] = None
        ):
        self.llm = llm
        self.retriever = retriever
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.n_results = n_results
        self.where = where
        self.where_document = where_document

    @abstractmethod    
    def invoke(self, query: str) -> List[Dict[str, Any]]:
        results = self._invoke(query, self.context_messages)
        return self.transform_to_documents(results)
    
    @abstractmethod
    def invoke_format_to_str(
        self,
        query: str
    ) -> str:
        """重写query,使用新query进行检索，返回格式化后的字符串"""
        results = self._invoke(
            query=query
        )
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        results_str = "\n\n".join([f"Document {index+1}: \n{result}" for index, result in enumerate(results['documents'][0])])
        page_content = results['documents'][0]
        metadatas = results['metadatas'][0]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas)

    def _build_contextual_query(
            self, 
            query: str,
            messages: List[Dict[str, Any]],
            use_llm: bool = True
        ) -> str:
        """构建新的query"""
        # 将messages转换为字符串
        if messages is None:
            messages = []
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
    
    @classmethod
    def transform_to_documents(
            cls,
            query_results: Dict[str, Any]
        ):
        """Transform the origin query results to a list of documents"""
        result = []
        documents = query_results['documents'][0]
        metadatas = query_results['metadatas'][0]
        
        for doc, meta in zip(documents, metadatas):
            result.append({
                'page_content': doc,
                'metadata': meta
            })
        
        return result
    
    def update_context_messages(
        self,
        messages: List[Dict[str, Any]],
    ):
        """更新messages"""
        self.context_messages = messages