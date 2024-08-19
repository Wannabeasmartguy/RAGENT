from loguru import logger
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from modules.retrievers.base import BaseRetriever, BaseContextualRetriever
from modules.llm.openai import OpenAILLM
from typing import List, Dict, Optional, Literal, Any, Coroutine


class ChromaRetriever(BaseRetriever):
    def __init__(
            self, 
            collection_name: str, 
            embedding_model: str, 
            device: Literal['mps', 'cuda', 'cpu'] = 'cpu'
        ):
        self.client = PersistentClient('./databases/knowledgebase')
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device=device
        )
        self.collection = self.client.get_collection(
            collection_name,
            embedding_function=self.embedding_model
        )

    def invoke(
            self, 
            query_texts: List[str], 
            n_results: int, 
            where: Optional[Dict] = None, 
            where_document: Optional[Dict] = None
        ) -> List[List[Dict[str,Any]]]:
        result =  self.collection.query(
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document
        )
        logger.info(f"Retrieved {len(result['documents'][0])} documents")
        return result
    
    def invoke_format_to_str(
            self, 
            query_texts: List[str], 
            n_results: int = 6, 
            where: Optional[Dict] = None, 
            where_document: Optional[Dict] = None
        ) -> str:
        """Format the results to a string"""
        results = self.invoke(
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document
        )
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        return "\n\n".join([f"Document {index+1}: \n{result}" for index, result in enumerate(results['documents'][0])])
    
    def ainvoke(self, query: str) -> Coroutine[Any, Any, List[Dict[str, Any]]]:
        return super().ainvoke(query)


class ChromaContextualRetriever(BaseContextualRetriever):
    def __init__(
            self, 
            collection_name: str, 
            embedding_model: str, 
            llm: OpenAILLM, 
            device: Literal['mps', 'cuda', 'cpu'] = 'cpu'
        ):
        super().__init__(llm, ChromaRetriever(collection_name=collection_name, embedding_model=embedding_model))
        self.client = PersistentClient('./databases/knowledgebase')
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
            device=device
        )
        self.collection = self.client.get_collection(
            collection_name,
            embedding_function=self.embedding_model
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