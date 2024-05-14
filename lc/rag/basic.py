from typing import Dict, List, Optional, Union, Literal, Any

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema.retriever import BaseRetriever

from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from lc.llm.openailike.completion import OpenAILikeLLM
from lc.chain.rerank import BgeRerank
from configs.knowledge_base_config import ChromaCollectionProcessor

import json

def create_lc_embedding_model(
    # embedding_config: Dict[str, Any],
    collection_name: str,
) -> AzureOpenAIEmbeddings | SentenceTransformerEmbeddings:
    # 读取本地的 embedding_config.json
    with open("embedding_config.json", "r") as f:
        embedding_config = json.load(f)
    
    if embedding_config[collection_name]["embedding_type"] == "openai":
        embedding_config_processed = embedding_config[collection_name]
        embedding_model = AzureOpenAIEmbeddings(
            openai_api_type=embedding_config_processed["api_type"],
            azure_endpoint=embedding_config_processed["base_url"],
            openai_api_key=embedding_config_processed["api_key"],
            openai_api_version=embedding_config_processed["api_version"],
            azure_deployment=embedding_config_processed["embedding_model_or_path"],
        )
    elif embedding_config[collection_name]["embedding_type"] == "huggingface":
        embedding_config_processed = embedding_config[collection_name]
        embedding_model = SentenceTransformerEmbeddings(
            model_name=embedding_config[collection_name]["embedding_model_or_path"]
        )

    return embedding_model
        

class ChromaCustomRetriever(BaseRetriever):
    """
    Used to create a custom retriever.
    """
    def __init__(
            self,
            collection_name: str,
            collection_processor: ChromaCollectionProcessor,
            n_results: int = 20,
        ):
        self.collection_name = collection_name
        self.collection_processor = collection_processor
        self.n_results = n_results

    def _get_relevant_documents(
            self, 
            query: str, 
            *, 
            run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
        # get relevant documents from collection
        # "query_response" is a dict
        query_response = self.collection_processor.search_docs(query,n_results=self.n_results)
        
        # "content" and "metadata" are lists
        content = query_response['documents'][0]
        metadata = query_response["metadatas"][0]
        
        # create a document list
        documents = []
        for i in range(len(content)):
            documents.append(Document(page_content=content[i], metadata=metadata[i]))
        
        return documents
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()        


def create_conversational_retrieval_system(
        llm,
        compressor, 
        collection_name: str,
        metadata: Dict,
        filter_goal: List = [0],
        filter_type: Literal['all', 'specific'] = "all",
        if_hybrid_retrieve: bool = False, 
        hybrid_retriever_weight: float = 0.5, 
        if_rerank: bool = False, 
    ):
    """
    创建并返回一个会话检索系统实例。

    Args:
        if_hybrid_retrieve (bool): 是否使用混合检索器。
        hybrid_retriever_weight (float): 混合检索器中BM25检索器的权重。
        if_rerank (bool): 是否进行重排。
        compressor : 用于实现重排序的压缩器。
        llm (object): 语言模型实例。
        filter_goal (list): 筛选目标。经过 find_source_paths() 找到的指定文档。
        filter_type (Literal['all', 'specific']): 筛选类型。

    Return:
        qa (object): 会话检索系统实例。
    """
    # 先创建embedding function
    embedding_function = create_lc_embedding_model(collection_name=collection_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    sparse_retrieve_kwargs = {
                'k': 6
            }
    if filter_type == "all":
        vec_retrieve_search_kwargs = {
            'k': 6
        }
        metadatas_in_sparse_retrieve = metadata
        
    elif filter_type == "specific":
        vec_retrieve_search_kwargs = {
            'k': 6,
            "filter":{
                "source":filter_goal[0]
            }
        }
        # metadatas_in_sparse_retrieve = filter_documents_by_source(vectorstore.get()["metadatas"],filter_goal)
        metadatas_in_sparse_retrieve = metadata.get("metadatas",[])

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=vec_retrieve_search_kwargs)

    if if_hybrid_retrieve:
        try:
            bm25_retriever = BM25Retriever.from_texts(
                vectorstore.get()["documents"],
                metadatas=metadatas_in_sparse_retrieve[0],
                kwargs=sparse_retrieve_kwargs
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=[hybrid_retriever_weight, 1 - hybrid_retriever_weight]
            )
        except Exception as e:
            raise "Could not import rank_bm25, please install with `pip install rank_bm25`" from e

    if if_rerank:
        if if_hybrid_retrieve:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        else:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            verbose=True,
            return_source_documents=True,
        )
    else:
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
        )

    return qa


class LCOpenAILikeRAGManager:
    """
    LCOpenAILikeRAGManager is a class that manages the RAG (Retrieval Augment Generation) process.
    It only suit for Langchain, based on the custom OpenAI-like LLM.
    """
    def __init__(
            self, 
            llm_config: Dict,
            collection: str
    ):
        """
        Initialize the LangchainRAGManager.
        
        Args:
            llm_config (Dict): The configuration of the LLM.
            collection (str): The name of the Chroma collection to use for RAG.
        """
        self.llm = OpenAILikeLLM(**llm_config)
        self.collection = collection

    def invoke(
            self, 
            prompt: str,
            chat_history: List[Dict],
            metadata: Dict,
            is_rerank: bool = False,
            is_hybrid_retrieve: bool = False,
    ) -> Dict:
        """
        Invoke the RAG process.
        
        Args:
            prompt (str): The prompt to use for RAG.
            
        Return:
            response: The generated response.
                Include "answer", and "content", which is the source files retrieved from the Chroma collection.
        """
        reranker = self.create_reranker(if_rerank=is_rerank)
        qa_system = create_conversational_retrieval_system(
            llm=self.llm,
            compressor=reranker,
            collection_name=self.collection,
            metadata=metadata,
            if_hybrid_retrieve=is_hybrid_retrieve,
            if_rerank=is_rerank,
        )
        
        response = qa_system.invoke({"question": prompt,"chat_history": chat_history})
        
        return response
    
    def create_reranker(
        self,
        if_rerank: bool = False,
    ) -> BgeRerank | None:
        """
        Create a BgeRerank object for reranking the retrieved documents.
        
        Args:
            if_rerank (bool): Whether to use reranking. Default is False.
            
        Return:
            reranker: The BgeRerank object.
        """
        if if_rerank:
            return BgeRerank()
        else:
            return None