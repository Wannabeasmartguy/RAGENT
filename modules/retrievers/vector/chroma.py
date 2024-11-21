import os
from dotenv import load_dotenv
from loguru import logger
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from modules.retrievers.base import BaseRetriever, BaseContextualRetriever
from modules.llm.openai import OpenAILLM
from typing import List, Dict, Optional, Literal, Any, Coroutine

load_dotenv(override=True)

class ChromaRetriever(BaseRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        embedding_type: Literal["openai", "aoai", "sentence_transformer"] = "sentence_transformer",
        device: Literal["mps", "cuda", "cpu"] = "cpu",
        *,
        n_results: int = 6,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        knowledge_base_path: str = "./databases/knowledgebase",
        distance_threshold: Optional[float] = None,
    ):
        self.client = PersistentClient(path=knowledge_base_path)

        if embedding_type == "openai":
            self.embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embedding_model, api_key=os.getenv("OPENAI_API_KEY")
            )
        elif embedding_type == "aoai":
            self.embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embedding_model, api_key=os.getenv("AZURE_OAI_KEY"),api_base=os.getenv("AZURE_OAI_ENDPOINT"),api_type="azure", api_version=os.getenv("API_VERSION")
            )
        elif embedding_type == "sentence_transformer":
            self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model, device=device
        )
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}")
        
        self.collection = self.client.get_collection(
            collection_name, embedding_function=self.embedding_model
        )
        self.n_results = n_results
        self.where = where
        self.where_document = where_document
        self.distance_threshold = distance_threshold

    def invoke(
        self, 
        query: str,
    ) -> List[Dict[str, Any]]:
        results = self._invoke(query)
        results_unfiltered = self.transform_to_documents(results)
        if self.distance_threshold:
            results_filtered = [result for result in results_unfiltered if result["distance"] <= self.distance_threshold]
            logger.info(f"Threshold is set as {self.distance_threshold}, filtered {len(results_unfiltered) - len(results_filtered)} documents")
            return results_filtered
        return results_unfiltered

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

        results_unfiltered = self.transform_to_documents(results)
        if self.distance_threshold:
            results_filtered = [result for result in results_unfiltered if result["distance"] <= self.distance_threshold]
            logger.info(f"Threshold is set as {self.distance_threshold}, filtered {len(results_unfiltered) - len(results_filtered)} documents")
        else:
            results_filtered = results_unfiltered

        results_str = "\n\n".join(
            [
                f"Document {index+1}: \n{result['page_content']}"
                for index, result in enumerate(results_filtered)
            ]
        )
        page_content = [result["page_content"] for result in results_filtered]
        metadatas = [result["metadatas"] for result in results_filtered]
        distances = [result["distance"] for result in results_filtered]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas, distances=distances)

    def ainvoke(self, query: str) -> Coroutine[Any, Any, List[Dict[str, Any]]]:
        return super().ainvoke(query)

    @classmethod
    def transform_to_documents(cls, query_results: Dict[str, Any]):
        """Transform the origin query results to a list of documents"""
        result = []
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            result.append(
                {
                    "page_content": doc,
                    "metadatas": meta,
                    "distance": dist
                }
            )

        return result
    
    def update_parameters(
        self,
        n_results: Optional[int] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        distance_threshold: Optional[float] = None,
    ):
        if n_results is not None:
            self.n_results = n_results
        if where is not None:
            self.where = where
        if where_document is not None:
            self.where_document = where_document
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold


class ChromaContextualRetriever(BaseContextualRetriever):
    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        llm: OpenAILLM,
        embedding_type: Literal["openai", "aoai", "sentence_transformer"] = "sentence_transformer",
        device: Literal["mps", "cuda", "cpu"] = "cpu",
        *,
        rewrite_by_llm: bool = True,
        n_results: int = 6,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
        knowledge_base_path: str = "./databases/knowledgebase",
        distance_threshold: Optional[float] = None,
    ):
        super().__init__(
            llm,
            ChromaRetriever(
                collection_name=collection_name,
                embedding_model=embedding_model,
                embedding_type=embedding_type,
                device=device,
                n_results=n_results,
                where=where,
                where_document=where_document,
                knowledge_base_path=knowledge_base_path,
                distance_threshold=distance_threshold,
            ),
        )
        self.rewrite_by_llm = rewrite_by_llm

    def invoke(self, query: str) -> List[Dict[str, Any]]:
        results = self._invoke(query)
        if self.retriever.distance_threshold:
            results_filtered = [result for result in results if result["distance"] <= self.retriever.distance_threshold]
            logger.info(f"Threshold is set as {self.retriever.distance_threshold}, filtered {len(results) - len(results_filtered)} documents")
        else:
            results_filtered = results
        return self.transform_to_documents(results_filtered)

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

        results_unfiltered = self.retriever.transform_to_documents(results)
        if self.retriever.distance_threshold:
            results_filtered = [result for result in results_unfiltered if result["distance"] <= self.retriever.distance_threshold]
            logger.info(f"Threshold is set as {self.retriever.distance_threshold}, filtered {len(results_unfiltered) - len(results_filtered)} documents")
        else:
            results_filtered = results_unfiltered

        results_str = "\n\n".join(
            [
                f"Document {index+1}: \n{result['page_content']}"
                for index, result in enumerate(results_filtered)
            ]
        )
        page_content = [result["page_content"] for result in results_filtered]
        metadatas = [result["metadatas"] for result in results_filtered]
        distances = [result["distance"] for result in results_filtered]
        return dict(result=results_str, page_content=page_content, metadatas=metadatas, distances=distances)

    @classmethod
    def transform_to_documents(cls, query_results: Dict[str, Any]):
        """Transform the origin query results to a list of documents"""
        result = []
        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            result.append(
                {
                    "page_content": doc,
                    "metadatas": meta,
                    "distance": dist
                }
            )

        return result
