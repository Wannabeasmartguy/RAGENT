import os
import json

from typing import Literal, List, Dict

import streamlit as st

from huggingface_hub import snapshot_download
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain_core.documents.base import Document

from utils.text_splitter.text_splitter_utils import simplify_filename

from configs.basic_config import I18nAuto
from api.dependency import APIRequestHandler
from api.routers.knowledgebase import EmbeddingModelConfig

i18n = I18nAuto()

requesthandler = APIRequestHandler("localhost", 8000)

DEFAULT_COLLECTION_NAME = "default_collection"


def create_embedding_model_config(
    embedding_model_type: Literal["openai", "huggingface"],
    embedding_model_name_or_path: str,
    **openai_kwargs
):
    if embedding_model_type == "openai":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="openai",
            embedding_model_or_path=embedding_model_name_or_path,
            api_key=openai_kwargs.get("api_key"),
            api_type=openai_kwargs.get("api_type"),
            base_url=openai_kwargs.get("base_url"),
            api_version=openai_kwargs.get("api_version"),
        )
    elif embedding_model_type == "huggingface":
        embedding_model_config = EmbeddingModelConfig(
            embedding_type="huggingface",
            embedding_model_or_path=embedding_model_name_or_path,
        )
    
    return embedding_model_config


def list_all_knowledgebase_collections() -> List[str]:
    """
    List all knowledgebase collections.

    Returns:
        List[str]: A list of collection names.
    """
    response = requesthandler.get("/knowledgebase/list-knowledge-bases")
    return response


def create_knowledgebase_collection(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
) -> None:
    """
    Create a knowledgebase collection.

    Args:
        collection_name (str): The name of the collection.
        embedding_model_type (str): The type of the embedding model.
        embedding_model_name_or_path (str): The name or path of the embedding model.
        openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
    """
    # 根据embedding_model_type选择不同的embedding模型配置
    embedding_model_config = create_embedding_model_config(
        embedding_model_type, 
        embedding_model_name_or_path, 
        **openai_kwargs
    )
    
    # 创建请求
    requesthandler.post(
        "/knowledgebase/create-knowledge-base",
        data=embedding_model_config.dict(),
        params={"name": collection_name},
    )


def delete_knowledgebase_collection(collection_name: str) -> None:
    """
    Delete a knowledgebase collection by name.

    Args:
        collection_name (str): The name of the collection.
    """
    requesthandler.post(
        "/knowledgebase/list-knowledge-bases",
        params={"name": collection_name},
    )


def list_collection_all_filechunks_content(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
    ) -> List[str]:
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type, 
        embedding_model_name_or_path, 
        **openai_kwargs
    )

    response = requesthandler.post(
        "/knowledgebase/list-all-files",
        data=embedding_model_config.dict(),
        params={"name": collection_name},
    )
    return response


def list_all_filechunks_in_detail(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
) -> Dict:
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )

    response = requesthandler.post(
        "/knowledgebase/list-all-files-in-detail",
        data=embedding_model_config.dict(),
        params={"name": collection_name}
    )
    return response


def list_all_filechunks_content_by_file_name(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
) -> List[str]:
    """
    List all files content in a collection by file name.
    
    Args:
        collection_name (str): The name of the collection.
        embedding_model_type (str): The type of the embedding model.
        embedding_model_name_or_path (str): The name or path of the embedding model.
        openai_kwargs (dict): Additional keyword arguments for the OpenAI embedding model.
        
    Returns:
        List[str]: A list of file content.
    """
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )

    response = requesthandler.post(
        "/knowledgebase/list-all-files-metadata-name",
        data=embedding_model_config.dict(),
        params={"name": collection_name}
    )
    return response


def search_docs(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        query:str | List[str],
        n_results:int,
        **openai_kwargs,
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )

    response = requesthandler.post(
        "/knowledgebase/search-docs",
        data={
            "query": query,
            "embedding_config": embedding_model_config.dict(),
        },
        params={"name": collection_name, "n_results": n_results}
    )
    return response


def add_documents(
        collection_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        documents:List[Document],
        **openai_kwargs,
        
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )
    
    requesthandler.post(
        "/knowledgebase/add-docs",
        data={
            "documents": documents,
            "embedding_config": embedding_model_config.dict(),
        },
        params={"name": collection_name}
    )


def delete_documents_from_same_metadata(
        collection_name: str,
        files_name: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )
    
    requesthandler.post(
        "/knowledgebase/delete-whole-file-in-collection",
        data=embedding_model_config.dict(),
        params={"name": collection_name, "files_name": files_name}
    )


def delete_specific_documents(
        collection_name: str,
        chunk_document_content: str,
        embedding_model_type:Literal["openai","huggingface"],
        embedding_model_name_or_path:str,
        **openai_kwargs,
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
    embedding_model_config = create_embedding_model_config(
        embedding_model_type,
        embedding_model_name_or_path,
        **openai_kwargs
    )
    
    requesthandler.post(
        "/knowledgebase/delete-specific-splitted-document",
        data=embedding_model_config.dict(),
        params={"name": collection_name, "chunk_document_content": chunk_document_content}
    )