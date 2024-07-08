import os
import json
import uuid

from typing import Literal, Optional, List, Dict
from abc import ABC, abstractmethod

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from huggingface_hub import snapshot_download
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_core.documents.base import Document

from utils.text_splitter.text_splitter_utils import simplify_filename

from configs.basic_config import I18nAuto, KNOWLEDGE_BASE_DIR
from api.dependency import APIRequestHandler
from api.routers.knowledgebase import (
    EmbeddingModelConfig, 
)
from model.config.embeddings import EmbeddingConfiguration, CollectionEmbeddingConfiguration

i18n = I18nAuto()

requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT",8000))

DEFAULT_COLLECTION_NAME = "default_collection"


class ChromaVectorStoreProcessStrategy(ABC):
    @abstractmethod
    def model_dir_verify(self, model_name_or_path: str) -> None:
        pass

    @abstractmethod
    def download_model(self, model_name_or_path: str, repo_id: str) -> None:
        pass

    @abstractmethod
    def list_all_knowledgebase_collections(self,) -> List[str]:
        pass

    @abstractmethod
    def create_knowledgebase_collection(
        self,
        collection_name: str,
        embedding_model_type: Literal["openai", "huggingface"],
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> None:
        pass

    @abstractmethod
    def delete_knowledgebase_collection(
        self,
        collection_name: str,
    ) -> None:
        pass


class BaseProcessStrategy(ABC):
    @abstractmethod
    def list_collection_all_filechunks_content(
            collection_name: str,
            embedding_model_type: Literal["openai", "huggingface"],
            embedding_model_name_or_path: str,
            **openai_kwargs,
        ) -> List[str]:
        pass
    
    @abstractmethod
    def list_all_filechunks_in_detail(
        collection_name: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> Dict:
        pass

    @abstractmethod
    def list_all_filechunks_metadata_name(
        collection_name: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> Dict:
        pass

    @abstractmethod
    def add_documents(
        collection_name: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        documents:List[Document],
        **openai_kwargs,
    ) -> None:
        pass

    @abstractmethod
    def delete_documents_from_same_metadata(
        collection_name: str,
        files_name: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> None:
        pass

    @abstractmethod
    def delete_specific_documents(
        collection_name: str,
        chunk_document_content: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> None:
        pass


class BaseChromaInitEmbeddingConfig:
    @classmethod
    def _list_chroma_collections(cls) -> List[str]:
        """
        List all knowledgebase collections.

        Returns:
            List[str]: A list of collection names.
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)
        raw_collections = client.list_collections()
        collections = [collection.name for collection in raw_collections]
        return collections

    @classmethod
    def _create_embedding_model(
        cls,
        embedding_config: EmbeddingConfiguration
    ) -> chromadb.EmbeddingFunction :
        '''根据embedding_type和embedding_model选择相应的模型'''
        if embedding_config.embedding_type == "openai":
            embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embedding_config.embedding_model_name_or_path,
                api_key=embedding_config.api_key,
                api_base=embedding_config.base_url,
                api_type=embedding_config.api_type,
                api_version=embedding_config.api_version
            )
        elif embedding_config.embedding_type == "huggingface":
            try:
                embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_config.embedding_model_name_or_path
                )
            except OSError:
                raise ValueError("Huggingface model not found, please use 'Local embedding model download' to download the model")
        else:
            raise ValueError("Unsupported embedding type")
        
        return embedding_model

    @classmethod
    def _get_chroma_specific_collection(cls, name: str, embedding_config: EmbeddingConfiguration) -> chromadb.Collection:
        '''
        获取知识库的特定 collection
        
        Args:
            name (str): 知识库某个 collection 的名称
            
        Returns:
            chromadb.Collection: 知识库中名称为 name 的 collection
        '''
        embedding_model = cls._create_embedding_model(embedding_config)
        knowledgebase_collections = cls._list_chroma_collections()
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)
        
        # Check if collection exists
        if name not in knowledgebase_collections:
            raise ValueError("Collection does not exist")
        
        collection = client.get_collection(name=name, embedding_function=embedding_model)
        return collection


def create_embedding_model_config(
    embedding_model_type: Literal["openai", "huggingface"],
    embedding_model_name_or_path: str,
    **openai_kwargs
):
    if embedding_model_type == "openai":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="openai",
            embedding_model_name_or_path=embedding_model_name_or_path,
            api_key=openai_kwargs.get("api_key"),
            api_type=openai_kwargs.get("api_type"),
            base_url=openai_kwargs.get("base_url"),
            api_version=openai_kwargs.get("api_version"),
        )
    elif embedding_model_type == "huggingface":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="huggingface",
            embedding_model_name_or_path=embedding_model_name_or_path,
        )
    
    return embedding_model_config


class ChromaVectorStoreProcessor(ChromaVectorStoreProcessStrategy):
    """Chroma向量存储处理器，用于在*使用FastAPI后端时*，使用request向后端请求以处理向量存储。"""
    embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")
    def __init__(
        self,
        embedding_model_type: Literal["openai", "huggingface"],
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ):
        self.embedding_model_config = create_embedding_model_config(
            embedding_model_type,
            embedding_model_name_or_path,
            **openai_kwargs
        )

    
    def model_dir_verify(self, model_name_or_path: str):
        """
        Verify the model directory. If the model_name_or_path is a directory, it will be used as the model directory.
        If the model_name_or_path doesn't exist, it will be created.

        Args:
            model_name_or_path (str): The model name or path. It's a directory path, consists of "models_dir_path/model_name".
        """
        if not os.path.exists(model_name_or_path):
            # 创建模型目录
            # os.makedirs(model_name_or_path, exist_ok=True)

            create_info = i18n("You don't have this embed model yet. Please enter huggingface model 'repo_id' to download the model FIRST.")
            return create_info
        else:
            # 模型目录存在
            return None
    
    
    def download_model(self, model_name_or_path: str, repo_id: str) -> None:
        """
        Download the model from Hugging Face Hub.

        Args:
            model_name_or_path (str): The model name or path. It's a directory path, consists of "models_dir_path/model_name".
        """
        # 下载模型
        os.makedirs(model_name_or_path, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_name_or_path,
        )
        
        return i18n("Model downloaded successfully!")
    
    
    @st.cache_data
    def list_all_knowledgebase_collections(_self,counter: int) -> List[str]:
        """
        List all knowledgebase collections.

        Args:
            counter (int): No usage, just for update cache data.

        Returns:
            List[str]: A list of collection names.
        """
        response = requesthandler.get("/knowledgebase/list-knowledge-bases")
        return response

    
    def create_knowledgebase_collection(
            self,
            collection_name: str
    ) -> None:
        """
        Create a knowledgebase collection.

        Args:
            collection_name (str): The name of the collection.
            embedding_model_type (str): The type of the embedding model.
            embedding_model_name_or_path (str): The name or path of the embedding model.
            openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
        """
        # 创建请求
        requesthandler.post(
            "/knowledgebase/create-knowledge-base",
            data=self.embedding_model_config.dict(),
            params={"name": collection_name},
        )

        # 请求完成后，在 embedding_config.json 中更新该 collection 的全部信息
        # 格式为
        # {
        #     collection_name: self.embedding_model_config.dict()
        # }
        
        # 更新 embedding_config.json，如果没有该文件，则创建
        if not os.path.exists(self.embedding_config_file_path):
            with open(self.embedding_config_file_path, "w") as f:
                json.dump({collection_name: self.embedding_model_config.dict()}, f)
        else:
            with open(self.embedding_config_file_path, "r") as f:
                collections_embedding_config = json.load(f)
            collections_embedding_config[collection_name] = self.embedding_model_config.dict()
            with open(self.embedding_config_file_path, "w") as f:
                json.dump(collections_embedding_config, f, indent=4)
    

    def delete_knowledgebase_collection(self,collection_name:str) -> None:
        """
        Delete a knowledgebase collection by name.

        Args:
            collection_name (str): The name of the collection.
        """
        requesthandler.post(
            "/knowledgebase/delete-knowledge-base",
            data=None,
            params={"name": collection_name},
        )
        
        # 删除 embedding_config.json 中的该 collection 的信息
        with open(self.embedding_config_file_path, "r") as f:
            collections_embedding_config = json.load(f)
        del collections_embedding_config[collection_name]
        with open(self.embedding_config_file_path, "w") as f:
            json.dump(collections_embedding_config, f, indent=4)
            

class ChromaCollectionProcessor(BaseProcessStrategy):
    def __init__(
            self,
            collection_name: str,
            embedding_model_type: Literal["openai", "huggingface"],
            embedding_model_name_or_path: str,
            **openai_kwargs,
        ):
        self.collection_name = collection_name
        self.embedding_model_config = create_embedding_model_config(
            embedding_model_type,
            embedding_model_name_or_path,
            **openai_kwargs
        )


    def get_embedding_model_max_seq_len(self) -> int:
        """
        Get the max sequence length of the embedding model.

        Args:
            embedding_model_type (str): The type of the embedding model.
            embedding_model_name_or_path (str): The name or path of the embedding model.
            openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
        
        Returns:
            int: The max sequence length of the embedding model.
        """
        response = requesthandler.post(
            "/knowledgebase/get-max-seq-len",
            data=self.embedding_model_config.dict(),
        )
        return response


    def list_collection_all_filechunks_content(self) -> List[str]:
        """
        List all files content in a collection.

        Args:
            collection_name (str): The name of the collection.
            embedding_model_type (str): The type of the embedding model.
            embedding_model_name_or_path (str): The name or path of the embedding model.
            openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
        
        Returns:
            List[str]: A list of file content.
        """
        response = requesthandler.post(
            "/knowledgebase/list-all-files",
            data=self.embedding_model_config.dict(),
            params={"name": self.collection_name},
        )
        return response


    def list_all_filechunks_in_detail(self) -> Dict:
        """
        List all file chunks in a collection.

        Args:
            collection_name (str): The name of the collection.
            embedding_model_type (str): The type of the embedding model.
            embedding_model_name_or_path (str): The name or path of the embedding model.
            openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
        
        Returns:
            Dict: A dictionary of file chunks info, include "ids", "embeddings", "metadatas" and "documents".
        """
        response = requesthandler.post(
            "/knowledgebase/list-all-files-in-detail",
            data=self.embedding_model_config.dict(),
            params={"name": self.collection_name}
        )
        return response


    @st.cache_data
    def list_all_filechunks_metadata_name(_self,counter:int) -> List[str]:
        """
        List all files content in a collection by file name.
        
        Args:
            counter (int): No usage, just for update cache data.
            
        Returns:
            List[str]: A list of file content.
        """
        response = requesthandler.post(
            "/knowledgebase/list-all-files-metadata-name",
            data=_self.embedding_model_config.dict(),
            params={"name": _self.collection_name}
        )
        
        if "error" in response:
            return []
        else:
            return response


    def search_docs(
            self,
            query:str | List[str],
            n_results:int,
    ) -> Dict:
        """
        查询知识库中的文档，返回与查询语句最相似 n_results 个文档列表
        
        Args:
            collection_name (str): 知识库名称
            embedding_model_type (str): 嵌入模型类型
            embedding_model_name_or_path (str): 嵌入模型名称或路径
            query (str | List[str]): 查询的文本或列表
            n_results (int): 返回结果的数量
            openai_kwargs (dict): 选择openai作为嵌入模型来源时，传递给嵌入模型的额外参数
            
        Returns:
            Dict: 查询结果
                ids (List[List[str]]): 匹配文档的 ID
                distances (List[List[float]]): 匹配文档的向量距离
                metadata (List[List[Dict]]): 匹配文档的元数据
                embeddings : 匹配文档的嵌入向量
                documents (List[List[str]]): 匹配文档的文本内容
        """
        response = requesthandler.post(
            "/knowledgebase/search-docs",
            data={
                "query": query,
                "embedding_config": self.embedding_model_config.dict(),
            },
            params={"name": self.collection_name, "n_results": n_results}
        )
        return response


    def add_documents(
            self,
            documents:List[Document],          
    ) -> None:
        """
        向知识库中添加文档
        
        Args:
            collection_name (str): 知识库名称
            embedding_model_type (str): 嵌入模型类型
            embedding_model_name_or_path (str): 嵌入模型名称或路径
            documents (List[Document]): 要添加的文档列表
            openai_kwargs (dict): 选择openai作为嵌入模型来源时，传递给嵌入模型的额外参数
        """
        # Document 类无法被 json 序列化，需要将其转换为字典
        documents_dict = [dict(document) for document in documents]

        requesthandler.post(
            "/knowledgebase/add-docs",
            data={
                "documents": documents_dict,
                "embedding_config": self.embedding_model_config.dict(),
            },
            params={"name": self.collection_name}
        )


    def delete_documents_from_same_metadata(
            self,
            files_name: str,
    ) -> None:
        """
        从知识库中删除来自于同一个相同元文件的文档块

        Args:
            collection_name (str): 知识库名称
            files_name (str): 元文件名称
            embedding_model_type (str): 嵌入模型类型
            embedding_model_name_or_path (str): 嵌入模型名称或路径
            openai_kwargs (dict): 选择openai作为嵌入模型来源时，传递给嵌入模型的额外参数
            
        """
        requesthandler.post(
            "/knowledgebase/delete-whole-file-in-collection",
            data=self.embedding_model_config.dict(),
            params={"name": self.collection_name, "files_name": files_name}
        )


    def delete_specific_documents(
            self,
            chunk_document_content: str,
    ) -> None:
        """
        从知识库中删除特定的文档块

        Args:
            collection_name (str): 知识库名称
            chunk_document_content (str): 要删除的文档块内容
            embedding_model_type (str): 嵌入模型类型
            embedding_model_name_or_path (str): 嵌入模型名称或路径
            openai_kwargs (dict): 选择openai作为嵌入模型来源时，传递给嵌入模型的额外参数
            
        """
        requesthandler.post(
            "/knowledgebase/delete-specific-splitted-document",
            data=self.embedding_model_config.dict(),
            params={"name": self.collection_name, "chunk_document_content": chunk_document_content}
        )


class ChromaVectorStoreProcessorWithNoApi(BaseChromaInitEmbeddingConfig,ChromaVectorStoreProcessor):
    """Chroma向量存储处理器，用于在*不使用FastAPI后端时*，直接处理向量存储。"""
    def __init__(
        self,
        embedding_model_type: Literal["openai", "huggingface"],
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> EmbeddingConfiguration:
        if embedding_model_type == "openai":
            self.embedding_model_config = EmbeddingConfiguration(
                model_id=str(uuid.uuid4()),
                embedding_type="openai",
                embedding_model_name_or_path=embedding_model_name_or_path,
                api_key=openai_kwargs.get("api_key"),
                api_type=openai_kwargs.get("api_type"),
                base_url=openai_kwargs.get("base_url"),
                api_version=openai_kwargs.get("api_version"),
            )
        elif embedding_model_type == "huggingface":
            self.embedding_model_config = EmbeddingConfiguration(
                model_id=str(uuid.uuid4()),
                embedding_type="huggingface",
                embedding_model_name_or_path=embedding_model_name_or_path,
            )
        self.embedding_model: chromadb.EmbeddingFunction = self._create_embedding_model(self.embedding_model_config),
        self.knowledgebase_collections: List[str] = self._list_chroma_collections()

    @st.cache_data
    def list_all_knowledgebase_collections(_self,counter: Optional[int] = None) -> List[str]:
        """
        List all knowledgebase collections.

        Args:
            counter (int): No usage, just for update cache data.

        Returns:
            List[str]: A list of collection names.
        """
        collections = _self._list_chroma_collections()
        return collections

    def create_knowledgebase_collection(
        self,
        name: str,
    ) -> None:
        '''
        创建一个知识库 collection
        
        Args:
            name (str): 知识库名称
            
        Returns:
            None
        '''
        
        client = chromadb.PersistentClient(path=self.embedding_config_file_path)

        # Check if collection already exists
        if name in self.knowledgebase_collections:
            raise ValueError("Collection already exists")

        # Create collection
        client.create_collection(
            name=name,
            embedding_function=self.embedding_model
        )
    
    def delete_knowledgebase_collection(
        self,
        name: str,
    ) -> None:
        '''
        删除指定名称的 collection
        
        Args:
            name (str): 知识库名称
            knowledgebase_collections (List[str], optional): 知识库的所有 collection
            
        Returns:
            None
        '''
        knowledgebase_collections: List[str] = self._list_chroma_collections()
        client = chromadb.PersistentClient(path=self.embedding_config_file_path)

        # Check if collection exists
        if name not in knowledgebase_collections:
            raise ValueError("Collection does not exist")
        
        # Delete collection
        client.delete_collection(name=name)
    
class ChromaCollectionProcessorWithNoApi(BaseChromaInitEmbeddingConfig,ChromaCollectionProcessor):
    """Chroma向量存储处理器，用于在*不使用FastAPI后端时*，直接处理向量存储。"""
    def __init__(
        self,
        collection_name: str,
        embedding_model_id: str,
        embedding_model_type: Literal["openai", "huggingface"],
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ) -> EmbeddingConfiguration:
        if embedding_model_type == "openai":
            self.embedding_model_config = EmbeddingConfiguration(
                model_id=embedding_model_id,
                embedding_type="openai",
                embedding_model_name_or_path=embedding_model_name_or_path,
                api_key=openai_kwargs.get("api_key"),
                api_type=openai_kwargs.get("api_type"),
                base_url=openai_kwargs.get("base_url"),
                api_version=openai_kwargs.get("api_version"),
            )
        elif embedding_model_type == "huggingface":
            self.embedding_model_config = EmbeddingConfiguration(
                model_id=embedding_model_id,
                embedding_type="huggingface",
                embedding_model_name_or_path=embedding_model_name_or_path,
            )
        self.collection = self._get_chroma_specific_collection(collection_name,self.embedding_model_config)
        self.collection_name = collection_name
    
    def get_embedding_model_max_seq_len(self) -> int:
        embedding_model = self._create_embedding_model(self.embedding_model_config)
        if isinstance(embedding_model,embedding_functions.OpenAIEmbeddingFunction):
            return 1500
        if isinstance(embedding_model,embedding_functions.SentenceTransformerEmbeddingFunction):
            model_path = list(embedding_model.models.keys())[0]
            # model_name = model_path.split("/")[-1]
            return embedding_model.models[model_path].max_seq_length

    def list_collection_all_filechunks_content(self) -> List[str]:
        """
        List all filechunks content in the collection.
        
        Returns:
            List[str]: A list of filechunks content.
        """
        document_count = self.collection.count()  # 获取文档数量
        document_situation = self.collection.peek(limit=document_count)

        document_list = document_situation["documents"]
        return document_list

    def list_all_filechunks_in_detail(self) -> Dict:
        """
        List all file chunks in a collection.
        
        Returns:
            Dict: A dictionary of file chunks info, include "ids", "embeddings", "metadatas" and "documents".
        """
        document_count = self.collection.count()  # 获取文档数量
        document_situation = self.collection.peek(limit=document_count)

        return document_situation
    
    @st.cache_data
    def list_all_filechunks_metadata_name(_self,counter:int) -> List[str]:
        """
        List all files content in a collection by file name.
        
        Args:
            counter (int): No usage, just for update cache data.
            
        Returns:
            List[str]: A list of file content.
        """
        client_data = _self.collection.get()
        unique_sources = set(client_data['metadatas'][i]['source'] for i in range(len(client_data['metadatas'])))
        # Extract actual file names
        file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]
        
        return file_names

    def search_docs(
            self,
            query:str | List[str],
            n_results:int,
    ) -> Dict:
        """
        查询知识库中的文档，返回与查询语句最相似 n_results 个文档列表
        
        Args:
            query (str | List[str]): 查询的文本或列表
            n_results (int): 返回结果的数量
            
        Returns:
            Dict: 查询结果
                ids (List[List[str]]): 匹配文档的 ID
                distances (List[List[float]]): 匹配文档的向量距离
                metadata (List[List[Dict]]): 匹配文档的元数据
                embeddings : 匹配文档的嵌入向量
                documents (List[List[str]]): 匹配文档的文本内容
        """
        results = self.collection.query(
            query_texts=query,
            n_results=n_results,
        )
    
        return results

    def add_documents(
            self,
            documents:List[Document],          
    ) -> None:
        """
        向知识库中添加文档
        
        Args:
            documents (List[Document]): 要添加的文档列表
        """
        # Document 类无法被 json 序列化，需要将其转换为字典
        documents_dict = [dict(document) for document in documents]

        # 使用列表推导式构造符合要求的text和metadata list
        page_content = [doc["page_content"] for doc in documents_dict]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [str(uuid.uuid4()) for _ in page_content]

        # Add texts to collection
        self.collection.add(
            documents=page_content,
            metadatas=metadatas,
            ids=ids,
        )

    def delete_documents_from_same_metadata(
            self,
            files_name: str,
    ) -> None:
        """
        从知识库中删除来自于同一个相同元文件的文档块

        Args:
            files_name (str): 元文件名称
            
        """
        # Delete file from collection
        # Initialize an empty list to store the ids
        ids_for_target_file = []

        # Get the documents from the collection
        metadata = self.collection.get()
        
        # Loop over the metadata
        for i in range(len(metadata['metadatas'])):
            # Check if the source matches the target file
            # We only compare the last part of the path (the actual file name)
            if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == files_name:
                # If it matches, add the corresponding id to the list
                ids_for_target_file.append(metadata['ids'][i])

        self.collection.delete(ids=ids_for_target_file)
    
    def delete_specific_documents(
            self,
            chunk_document_content: str,
    ) -> None:
        """
        从知识库中删除特定的文档块

        Args:
            chunk_document_content (str): 要删除的文档块内容
            
        """
        self.collection.delete(where_document={"$contains": chunk_document_content})