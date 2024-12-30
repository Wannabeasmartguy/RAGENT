import os
import json
import uuid

from typing import Literal, Optional, List, Dict
from abc import ABC, abstractmethod
from deprecated import deprecated

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from huggingface_hub import snapshot_download
from langchain_core.documents.base import Document
from loguru import logger

from core.basic_config import I18nAuto
from config.constants.paths import KNOWLEDGE_BASE_DIR
from api.routers.knowledgebase import (
    EmbeddingModelConfig,
)
from core.models.embeddings import (
    EmbeddingModelConfiguration,
    KnowledgeBaseConfiguration,
    GlobalSettings,
    EmbeddingConfiguration,
)


EMBEDDING_CONFIG_FILE_PATH = os.path.join("dynamic_configs", "embedding_config.json")

DEFAULT_COLLECTION_NAME = "default_collection"


class ChromaVectorStoreProcessStrategy(ABC):
    @abstractmethod
    def model_dir_verify(self, model_name_or_path: str) -> None:
        pass

    @abstractmethod
    def download_model(self, model_name_or_path: str, repo_id: str) -> None:
        pass

    @abstractmethod
    def list_all_knowledgebase_collections(
        self,
    ) -> List[str]:
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
        documents: List[Document],
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
        列出所有知识库集合。

        返回：
            List[str]：集合名称列表。
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)
        raw_collections = client.list_collections()
        collections = [collection.name for collection in raw_collections]
        return collections

    @abstractmethod
    def _create_embedding_model(
        cls, embedding_config: EmbeddingModelConfiguration
    ) -> chromadb.EmbeddingFunction:
        """根据embedding_type和embedding_model选择相应的模型"""
        if embedding_config.embedding_type == "openai":
            embedding_model = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embedding_config.embedding_model_name_or_path,
                api_key=embedding_config.api_key,
                api_base=embedding_config.base_url,
                api_type=embedding_config.api_type,
                api_version=embedding_config.api_version,
            )
        elif embedding_config.embedding_type == "huggingface":
            try:
                local_model_path = os.path.join(
                    "embeddings", embedding_config.embedding_model_name_or_path
                )
                if os.path.exists(local_model_path):
                    embedding_model = (
                        embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name=local_model_path
                        )
                    )
                else:
                    raise ValueError(
                        f"Local model path {local_model_path} does not exist, please ensure the model has been downloaded."
                    )
            except Exception as e:
                raise ValueError(
                    f"Huggingface model not found, please use 'Local embedding model download' to download the model"
                )
        else:
            raise ValueError("Unsupported embedding type")

        return embedding_model

    @classmethod
    def _get_chroma_specific_collection(
        cls, name: str, embedding_model: chromadb.EmbeddingFunction
    ) -> chromadb.Collection:
        """
        获取知识库的特定 collection

        参数：
            name (str): 知识库某个 collection 的名称
            embedding_model (chromadb.EmbeddingFunction): 嵌入模型

        返回：
            chromadb.Collection: 知识库中名称为 name 的 collection
        """
        knowledgebase_collections = cls._list_chroma_collections()
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)

        # 检查集合是否存在
        if name not in knowledgebase_collections:
            return None

        collection = client.get_collection(
            name=name, embedding_function=embedding_model
        )

        return collection


def create_embedding_model_config(
    embedding_model_type: Literal["openai", "aoai", "sentence_transformer"],
    embedding_model_name_or_path: str,
    **openai_kwargs,
):
    if embedding_model_type == "openai":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="openai",
            embedding_model_name_or_path=embedding_model_name_or_path,
            api_key=openai_kwargs.get("api_key"),
        )
    elif embedding_model_type == "aoai":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="aoai",
            embedding_model_name_or_path=embedding_model_name_or_path,
            api_key=openai_kwargs.get("api_key"),
            api_type=openai_kwargs.get("api_type"),
            base_url=openai_kwargs.get("base_url"),
            api_version=openai_kwargs.get("api_version"),
        )
    elif embedding_model_type == "sentence_transformer":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="sentence_transformer",
            embedding_model_name_or_path=embedding_model_name_or_path,
        )

    return embedding_model_config


class ChromaVectorStoreProcessorWithNoApi(BaseChromaInitEmbeddingConfig):
    """Chroma向量存储处理器，用于在*不使用FastAPI后端时*，直接处理向量存储。"""

    def __init__(
        self,
        embedding_model_type: Literal["openai", "aoai", "sentence_transformer"],
        embedding_model_name_or_path: str,
        **openai_kwargs,
    ):
        """初始化向量存储处理器
        
        Args:
            embedding_model_type: 嵌入模型类型，支持 "openai"、"aoai" 或 "sentence_transformer"
            embedding_model_name_or_path: 模型名称或路径
            **openai_kwargs: OpenAI相关的额外参数(api_key, base_url等)
        """
        # 1. 保存基本模型信息
        self.embedding_model_type = embedding_model_type
        self.embedding_model_name_or_path = embedding_model_name_or_path

        # 2. 创建模型配置信息
        model_id = str(uuid.uuid4())
        embedding_model = self._create_model_config(
            model_id, 
            embedding_model_type,
            embedding_model_name_or_path,
            openai_kwargs
        )

        # 3. 尝试加载已有模型信息，或创建新的嵌入模型配置信息
        self.embedding_config = self._load_or_create_embedding_config(
            model_id, 
            embedding_model
        )

        # 4. 初始化嵌入模型和知识库集合
        self.embedding_model = self._create_embedding_model(embedding_model)
        self.knowledgebase_collections = self._list_chroma_collections()

    def _create_model_config(
        self, 
        model_id: str,
        embedding_model_type: str,
        embedding_model_name_or_path: str,
        openai_kwargs: dict
    ) -> EmbeddingModelConfiguration:
        """
        创建模型配置，如果新增模型类型，请在此处添加
        
        Args:
            model_id (str): 模型ID
            embedding_model_type (str): 嵌入模型类型
            embedding_model_name_or_path (str): 模型名称或路径
            openai_kwargs (dict): OpenAI相关的额外参数(api_key, base_url等)

        Returns:
            EmbeddingModelConfiguration: 模型配置
        """
        model_configs = {
            "aoai": {
                "name": f"Azure OpenAI Embedding Model {model_id[:8]}",
                "embedding_type": "aoai",
                "extra_params": {
                    "api_key": openai_kwargs.get("api_key"),
                }
            },
            "openai": {
                "name": f"OpenAI Embedding Model {model_id[:8]}",
                "embedding_type": "openai",
                "extra_params": {
                    "api_key": openai_kwargs.get("api_key"),
                    "base_url": openai_kwargs.get("base_url"),
                }
            },
            "sentence_transformer": {
                "name": f"Sentence Transformer Embedding Model {model_id[:8]}",
                "embedding_type": "sentence_transformer",
                "extra_params": {}
            }
        }

        if embedding_model_type not in model_configs:
            raise ValueError("Unsupported embedding type")

        config = model_configs[embedding_model_type]
        return EmbeddingModelConfiguration(
            id=model_id,
            name=config["name"],
            embedding_type=config["embedding_type"],
            embedding_model_name_or_path=embedding_model_name_or_path,
            **config["extra_params"]
        )

    def _load_or_create_embedding_config(
        self,
        model_id: str,
        embedding_model: EmbeddingModelConfiguration
    ) -> EmbeddingConfiguration:
        """
        尝试加载已有嵌入配置，如果配置不存在，则创建新的嵌入配置
        """
        if os.path.exists(EMBEDDING_CONFIG_FILE_PATH):
            # 加载已有配置
            try:
                with open(EMBEDDING_CONFIG_FILE_PATH, "r") as f:
                    return EmbeddingConfiguration(**json.load(f))
            except Exception as e:
                logger.error(f"Error loading embedding config: {e}")
                logger.info(f"Creating new embedding config for {model_id} due to loading error")

        # 创建新的配置
        from datetime import datetime
        return EmbeddingConfiguration(
            global_settings=GlobalSettings(default_model=model_id),
            models=[embedding_model],
            knowledge_bases=[],
            user_id=None,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

    @classmethod
    def download_model(cls, model_name_or_path: str, repo_id: str) -> str:
        """
        从Hugging Face Hub下载模型。

        Args:
            model_name_or_path (str): 模型名称或路径。这是一个目录路径，由"models_dir_path/model_name"组成。
            repo_id (str): Hugging Face Hub上的仓库ID。

        Returns:
            str: 下载状态的消息。

        Raises:
            ValueError: 如果model_name_or_path或repo_id为空。
            Exception: 下载过程中发生的任何其他错误。
        """
        if not model_name_or_path or not repo_id:
            raise ValueError("model_name_or_path和repo_id不能为空")

        try:
            os.makedirs(model_name_or_path, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_name_or_path,
            )
            return "Model downloaded successfully!"
        except Exception as e:
            error_message = f"Download model error: {str(e)}"
            logger.error(error_message)
            return error_message

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

            create_info = "You don't have this embed model yet. Please enter huggingface model 'repo_id' to download the model FIRST."
            return create_info
        else:
            # 模型目录存在
            return None

    @classmethod
    def _list_chroma_collections(cls) -> Dict[str, str]:
        """
        列出所有知识库集合。

        Returns:
            Dict[str, str]：键为用户指定的collection_name，值为内部使用的collection_id。
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)
        raw_collections = client.list_collections()
        collections = {
            collection.metadata.get(
                "user_collection_name", collection.name
            ): collection.name
            for collection in raw_collections
        }
        return collections

    def create_knowledgebase_collection(
        self,
        collection_name: str,
        hnsw_space: Literal["cosine", "l2"] = "cosine",
    ) -> None:
        """
        创建一个知识库 collection

        Args:
            collection_name (str): 用户指定的知识库名称

        Returns:
            None
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)

        # 检查集合是否已存在
        if collection_name in self.knowledgebase_collections:
            raise ValueError("Collection already exists")

        # 生成内部使用的collection_id
        collection_id = str(uuid.uuid4())

        # 创建集合，使用collection_id作为实际的collection名称，并在metadata中存储用户指定的名称
        client.create_collection(
            name=collection_id,
            embedding_function=self.embedding_model,
            metadata={
                "user_collection_name": collection_name,
                "hnsw:space": hnsw_space,
            },
        )

        # 获取当前使用的嵌入模型的ID
        current_model_id = self._get_current_model_id()

        # 添加新的知识库到配置中
        new_kb = KnowledgeBaseConfiguration(
            id=collection_id, name=collection_name, embedding_model_id=current_model_id
        )
        self.embedding_config.knowledge_bases.append(new_kb)

        # 更新内部的knowledgebase_collections字典
        self.knowledgebase_collections[collection_name] = collection_id

        # 更新配置文件
        self._update_embedding_config()

    def delete_knowledgebase_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        删除指定名称的 collection

        Args:
            collection_name (str): 用户指定的知识库名称

        Returns:
            None
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)

        # 检查集合是否存在
        if collection_name not in self.knowledgebase_collections:
            raise ValueError("Collection does not exist")

        collection_id = self.knowledgebase_collections[collection_name]

        # 删除collection
        client.delete_collection(name=collection_id)

        # 从配置中移除知识库
        self.embedding_config.knowledge_bases = [
            kb
            for kb in self.embedding_config.knowledge_bases
            if kb.name != collection_name
        ]

        # 从内部字典中移除
        del self.knowledgebase_collections[collection_name]

        # 更新配置文件
        self._update_embedding_config()

    def _update_embedding_config(self):
        """更新嵌入配置文件"""
        with open(EMBEDDING_CONFIG_FILE_PATH, "w") as f:
            json.dump(self.embedding_config.serializable_dict(), f, indent=2)

    def _create_embedding_model(
        self, model_config: EmbeddingModelConfiguration
    ) -> chromadb.EmbeddingFunction:
        """根据模型配置创建嵌入模型"""
        if model_config.embedding_type == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_config.embedding_model_name_or_path,
                api_key=model_config.api_key,
            )
        elif model_config.embedding_type == "aoai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_config.embedding_model_name_or_path,
                api_key=model_config.api_key,
                api_base=model_config.base_url,
                api_type=model_config.api_type,
                api_version=model_config.api_version,
            )
        elif model_config.embedding_type == "sentence_transformer":
            local_model_path = os.path.join(
                "embeddings", model_config.embedding_model_name_or_path
            )
            if os.path.exists(local_model_path):
                return embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=local_model_path
                )
            else:
                raise ValueError(
                    "Huggingface model not found, please use 'Local embedding model download' to download the model"
                )
        else:
            raise ValueError("Unsupported embedding type")

    def _get_current_model_id(self):
        """
        获取当前使用的嵌入模型的ID

        Returns:
            str: 当前模型的ID
        """
        for model in self.embedding_config.models:
            if (
                model.embedding_type == self.embedding_model_type
                and model.embedding_model_name_or_path
                == self.embedding_model_name_or_path
            ):
                return model.id

        # 如果没有找到匹配的模型，使用默认模型ID
        return self.embedding_config.global_settings.default_model


class ChromaCollectionProcessorWithNoApi(BaseChromaInitEmbeddingConfig):
    """Chroma向量存储处理器，用于在*不使用FastAPI后端时*，直接处理向量存储。"""

    def __init__(
        self,
        collection_name: str,
        embedding_config: EmbeddingConfiguration,
        embedding_model_id: str,
    ):
        self.embedding_config = embedding_config
        self.embedding_model = self._get_embedding_model(embedding_model_id)
        self.collection_name = collection_name
        self.collection_id = self._get_collection_id(collection_name)
        self.collection = self._get_chroma_specific_collection(
            self.collection_id, self.embedding_model
        )

    def _get_collection_id(self, collection_name: str) -> str:
        """
        根据用户指定的collection_name获取对应的内部collection_id

        Args:
            collection_name (str): 用户指定的collection名称

        Returns:
            str: 对应的内部collection_id

        Raises:
            ValueError: 如果找不到对应的collection
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)
        collections = client.list_collections()
        for collection in collections:
            if collection.metadata.get("user_collection_name") == collection_name:
                return collection.name
        raise ValueError(f"No collection found with name: {collection_name}")

    @classmethod
    def _get_chroma_specific_collection(
        cls, collection_id: str, embedding_model: chromadb.EmbeddingFunction
    ) -> chromadb.Collection:
        """
        获取知识库的特定 collection

        参数：
            collection_id (str): 知识库某个 collection 的内部ID
            embedding_model (chromadb.EmbeddingFunction): 嵌入模型

        返回：
            chromadb.Collection: 知识库中ID为 collection_id 的 collection
        """
        client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_DIR)

        try:
            collection = client.get_collection(
                name=collection_id, embedding_function=embedding_model
            )
            return collection
        except ValueError:
            return None

    def _get_embedding_model(self, model_id: str) -> chromadb.EmbeddingFunction:
        model_config = next(
            (model for model in self.embedding_config.models if model.id == model_id),
            None,
        )
        if not model_config:
            raise ValueError(f"No embedding model found with id {model_id}")
        return self._create_embedding_model(model_config)

    def _create_embedding_model(
        self, model_config: EmbeddingModelConfiguration
    ) -> chromadb.EmbeddingFunction:
        if model_config.embedding_type == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_config.embedding_model_name_or_path,
                api_key=model_config.api_key,
                api_base=model_config.base_url,
            )
        elif model_config.embedding_type == "aoai":
            return embedding_functions.OpenAIEmbeddingFunction(
                model_name=model_config.embedding_model_name_or_path,
                api_key=model_config.api_key,
                api_base=model_config.base_url,
                api_type=model_config.api_type,
                api_version=model_config.api_version,
            )
        elif model_config.embedding_type == "sentence_transformer":
            try:
                # 使用SentenceTransformerEmbeddingFunction加载模型
                # 构造本地路径并传入
                local_model_path = os.path.join(
                    "embeddings", model_config.embedding_model_name_or_path
                )
                if os.path.exists(local_model_path):
                    return embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=local_model_path
                    )
                else:
                    raise ValueError(
                        f"Local model path {local_model_path} does not exist, please ensure the model has been downloaded."
                    )
            except Exception as e:
                raise ValueError(
                    f"Sentence Transformer model not found, please use 'Local embedding model download' to download the model"
                )
        else:
            raise ValueError("Unsupported embedding type")

    def get_embedding_model_max_seq_len(self) -> int:
        if isinstance(
            self.embedding_model, embedding_functions.OpenAIEmbeddingFunction
        ):
            return 1500
        if isinstance(
            self.embedding_model,
            embedding_functions.SentenceTransformerEmbeddingFunction,
        ):
            # 直接访问模型属性
            return self.embedding_model._model.max_seq_length

    def list_collection_all_filechunks_content(self) -> List[str]:
        """
        List all filechunks content in the collection.

        Returns:
            List[str]: A list of filechunks content.
        """
        document_count = self.collection.count()
        document_situation = self.collection.peek(limit=document_count)
        return document_situation["documents"]

    def list_all_filechunks_in_detail(self) -> Dict:
        """
        List all file chunks in a collection.

        Returns:
            Dict: A dictionary of file chunks info, include "ids", "embeddings", "metadatas" and "documents".
        """
        document_count = self.collection.count()
        return self.collection.peek(limit=document_count)

    @st.cache_data
    def list_all_filechunks_raw_metadata_name(_self, counter: int) -> List[str]:
        """
        List all file chunks raw metadata name in a collection.

        Returns:
            List[str]: A list of file chunks raw metadata name.
        """
        if _self.collection is None:
            return []
        client_data = _self.collection.get()
        unique_sources = set(
            metadata["source"] for metadata in client_data["metadatas"]
        )
        return unique_sources

    # @st.cache_data
    def list_all_filechunks_metadata_name(_self, counter: int) -> List[str]:
        """
        List all files content in a collection by file name.

        Args:
            counter (int): No usage, just for update cache data.

        Returns:
            List[str]: A list of file names.
        """
        if _self.collection is None:
            return []
        client_data = _self.collection.get()
        unique_sources = set(
            metadata["source"] for metadata in client_data["metadatas"]
        )
        return [source.split("/")[-1].split("\\")[-1] for source in unique_sources]

    def search_docs(
        self,
        query: str | List[str],
        n_results: int,
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
        return self.collection.query(
            query_texts=query,
            n_results=n_results,
        )

    def add_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        向知识库中添加文档

        Args:
            documents (List[Document]): 要添加的文档列表
        """
        page_content = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in page_content]

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
        metadata = self.collection.get()
        ids_for_target_file = [
            metadata["ids"][i]
            for i, meta in enumerate(metadata["metadatas"])
            if meta["source"].split("/")[-1].split("\\")[-1] == files_name
        ]
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
