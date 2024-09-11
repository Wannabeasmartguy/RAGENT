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
        device: Literal["mps", "cuda", "cpu"] = "cpu",
        *,
        n_results: int = 6,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        knowledge_base_path: str = "./databases/knowledgebase",
    ):
        self.client = PersistentClient(path=knowledge_base_path)
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device=device
        )
        self.collection = self.client.get_collection(
            collection_name, embedding_function=self.embedding_model
        )
        self.n_results = n_results
        self.where = where
        self.where_document = where_document

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        results = self._invoke(query)
        return self.transform_to_documents(results)

    def _invoke(
        self,
        query_texts: List[str],
    ) -> List[List[Dict[str, Any]]]:
        result = self.collection.query(
            query_texts=query_texts,
            n_results=self.n_results,
            where=self.where,
            where_document=self.where_document,
        )
        logger.info(f"Retrieved {len(result['documents'][0])} documents")
        return result

    def invoke_format_to_str(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Format the results to a string"""
        results = self._invoke(query_texts=[query])
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        results_str = "\n\n".join(
            [
                f"Document {index+1}: \n{result}"
                for index, result in enumerate(results["documents"][0])
            ]
        )
        page_content = results["documents"][0]
        metadatas = results["metadatas"][0]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas)

    def ainvoke(self, query: str) -> Coroutine[Any, Any, List[Dict[str, Any]]]:
        return super().ainvoke(query)

    @classmethod
    def transform_to_documents(cls, query_results: Dict[str, Any]):
        """Transform the origin query results to a list of documents"""
        result = []
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]

        for doc, meta in zip(documents, metadatas):
            result.append({"page_content": doc, "metadatas": meta})

        return result
    
    def update_parameters(
        self,
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ):
        if n_results is not None:
            self.n_results = n_results
        if where is not None:
            self.where = where
        if where_document is not None:
            self.where_document = where_document


class ChromaContextualRetriever(BaseContextualRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        llm: OpenAILLM,
        device: Literal["mps", "cuda", "cpu"] = "cpu",
        *,
        rewrite_by_llm: bool = True,
        n_results: int = 6,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ):
        super().__init__(
            llm,
            ChromaRetriever(
                collection_name=collection_name,
                embedding_model=embedding_model,
                device=device,
                n_results=n_results,
                where=where,
                where_document=where_document,
            ),
        )
        self.rewrite_by_llm = rewrite_by_llm

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        results = self._invoke(query)
        return self.transform_to_documents(results)

    def _invoke(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """重写query,使用新query进行检索"""
        new_query = self._build_contextual_query(
            query, self.context_messages, use_llm=self.rewrite_by_llm
        )
        logger.info(f"New query: {new_query}")
        return self.retriever._invoke(
            query_texts=[new_query],
        )

    def invoke_format_to_str(self, query: str) -> Dict[str, Any]:
        """重写query,使用新query进行检索，返回格式化后的字符串"""
        results = self._invoke(query=query)
        logger.info(f"Retrieved {len(results['documents'][0])} documents")
        results_str = "\n\n".join(
            [
                f"Document {index+1}: \n{result}"
                for index, result in enumerate(results["documents"][0])
            ]
        )
        page_content = results["documents"][0]
        metadatas = results["metadatas"][0]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas)

    @classmethod
    def transform_to_documents(cls, query_results: Dict[str, Any]):
        """Transform the origin query results to a list of documents"""
        result = []
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]

        for doc, meta in zip(documents, metadatas):
            result.append({"page_content": doc, "metadatas": meta})

        return result
