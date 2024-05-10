from fastapi import APIRouter, Depends
from fastapi import HTTPException
from pydantic import BaseModel

import openai
import chromadb
from chromadb.utils import embedding_functions

from autogen.oai import OpenAIWrapper

import langchain_community.vectorstores.chroma as lcchroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_core.embeddings import Embeddings as LC_Embeddings
from langchain_core.documents.base import Document
from huggingface_hub import snapshot_download

from typing import List, Dict, Literal


KNOWLEDGE_BASE_PATH = "./knowledgebase"


router = APIRouter(
    prefix="/knowledgebase",
    tags=["knowledgebase"],
    responses={404: {"description": "Not found"}},
)


class EmbeddingModelConfig(BaseModel):
    embedding_type: Literal["openai", "huggingface"]
    embedding_model_or_path: str
    api_key: str | None
    base_url: str | None
    api_type: str | None
    api_version: str | None


async def list_chroma_collections() -> List[str]:
    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)
    raw_collections = client.list_collections()
    collections = [collection.name for collection in raw_collections]
    return collections


async def create_embedding_model(
    embedding_config: EmbeddingModelConfig
) -> chromadb.EmbeddingFunction :
    '''根据embedding_type和embedding_model选择相应的模型'''
    if embedding_config.embedding_type == "openai":
        embedding_model = embedding_functions.OpenAIEmbeddingFunction(
            model_name=embedding_config.embedding_model_or_path,
            api_key=embedding_config.api_key,
            api_base=embedding_config.base_url,
            api_type=embedding_config.api_type,
            api_version=embedding_config.api_version
        )
    elif embedding_config.embedding_type == "huggingface":
        embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_config.embedding_model_or_path
        )
    else:
        raise ValueError("Unsupported embedding type")
    
    return embedding_model


async def create_lc_embedding_model(
    embedding_config: EmbeddingModelConfig,
) -> LC_Embeddings:
    '''根据embedding_type和embedding_model选择相应的模型'''
    if embedding_config.embedding_type == "openai":
        embedding_model = AzureOpenAIEmbeddings(
            openai_api_type=embedding_config.api_type,
            azure_endpoint=embedding_config.base_url,
            openai_api_key=embedding_config.api_key,
            openai_api_version=embedding_config.api_version,
            azure_deployment=embedding_config.embedding_model_or_path,
        )
    elif embedding_config.embedding_type == "huggingface":
        try:
            embedding_model = SentenceTransformerEmbeddings(
                model_name=embedding_config.embedding_model_or_path
            )
        except:
            raise HTTPException(
                status_code=400, 
                detail="Invalid embedding model, please check the model name or use huggingface model name and try again."
            )
    
    return embedding_model
    

async def get_chroma_specific_collection(
    name: str,
    embedding_model: chromadb.EmbeddingFunction = Depends(create_embedding_model),
    knowledgebase_collections: List[str] = Depends(list_chroma_collections)
) -> chromadb.Collection:
    '''
    获取知识库的特定 collection
    
    Args:
        name (str): 知识库某个 collection 的名称
        embedding_model (chromadb.EmbeddingFunction, optional): 知识库的特定 collection 的嵌入模型. 
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
        
    Returns:
        chromadb.Collection: 知识库中名称为 name 的 collection
    '''
    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)
    
    # Check if collection exists
    if name not in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection does not exist")
    
    # Search documents in collection
    collection = client.get_collection(name=name,embedding_function=embedding_model)
    return collection


@router.get("/list-knowledge-bases")
async def list_knowledge_bases(
    knowledgebase_collections: List[str] = Depends(list_chroma_collections)
) -> List[str]:
    '''
    获取知识库的所有 collection
    
    Args:
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
        
    Returns:
        List[str]: 知识库的所有 collection
    '''
    return knowledgebase_collections


@router.post("/create-knowledge-base",response_model_exclude_unset=True)
async def create_knowledge_base_collection(
    name: str,
    knowledgebase_collections: List[str] = Depends(list_chroma_collections),
    embedding_model: chromadb.EmbeddingFunction = Depends(create_embedding_model)
) -> None:
    '''
    创建一个知识库 collection
    
    Args:
        name (str): 知识库名称
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
        embedding_model (chromadb.EmbeddingFunction, optional): 嵌入模型
        
    Returns:
        None
    '''
    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)

    # Check if collection already exists
    if name in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection already exists")

    # Create collection
    client.create_collection(
        name=name,
        embedding_function=embedding_model
    )


@router.post("/delete-knowledge-base")
async def delete_knowledge_base_collection(
    name: str,
    knowledgebase_collections: List[str] = Depends(list_chroma_collections)
) -> None:
    '''
    删除指定名称的 collection
    
    Args:
        name (str): 知识库名称
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
        
    Returns:
        None
    '''
    client = chromadb.PersistentClient(path=KNOWLEDGE_BASE_PATH)

    # Check if collection exists
    if name not in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection does not exist")
    
    # Delete collection
    client.delete_collection(name=name)


@router.post("/list-all-files")
async def list_all_files_in_collection(
    collection: chromadb.Collection = Depends(get_chroma_specific_collection)
) -> List[str]:
    '''
    返回指定名称的 collection 中的所有文件块
    
    Args:
        name (str): 知识库名称
        knowledgebase_collections (List[str], optional): 知识库的所有 collection

    Returns:
        List[str]: 文件的具体内容列表
    '''
    document_count = collection.count()  # 获取文档数量
    document_situation = collection.peek(limit=document_count)

    document_list = document_situation["documents"]
    return document_list


@router.post("/list-all-files-metadata-name")
async def list_all_files_metadata_name(
    name: str,
    knowledgebase_collections: List[str] = Depends(list_chroma_collections),
    embedding_model: LC_Embeddings = Depends(create_lc_embedding_model)
) -> List[str]:
    '''
    返回指定名称的 collection 中文件 metadata 的文件名
    
    Args:
        name (str): 知识库名称
        knowledgebase_collections (List[str], optional): 知识库的所有 collection

    Returns:
        List[Dict]: 文件的具体内容列表，每个字典包含文件元数据和名称
    '''

    client = lcchroma.Chroma(
        collection_name=name,
        embedding_function=embedding_model,
        persist_directory=KNOWLEDGE_BASE_PATH
    )
    
    # Check if collection exists
    if name not in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection does not exist")
    
    client_data = client.get()
    unique_sources = set(client_data['metadatas'][i]['source'] for i in range(len(client_data['metadatas'])))
    # Extract actual file names
    file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]
    
    return file_names


@router.post("/search-docs")
async def query_docs_in_collection(
    query: str | List[str],
    n_results: int = 10,
    collection: chromadb.Collection = Depends(get_chroma_specific_collection)
) -> Dict:
    '''
    查询知识库中的文档，返回与查询语句最相似 n_results 个文档列表

    Args:
        name (str): 知识库名称
        query (str | List[str]): 查询的文本
        embedding_model (chromadb.EmbeddingFunction): 嵌入模型
        n_results (int, optional): 返回结果的数量. Defaults to 10.
        knowledgebase_collections (List[str], optional): 知识库的所有 collection

    Returns:
        Dict: 查询结果
            ids (List[List[str]]): 匹配文档的 ID
            distances (List[List[float]]): 匹配文档的向量距离
            metadata (List[List[Dict]]): 匹配文档的元数据
            embeddings : 匹配文档的嵌入向量
            documents (List[List[str]]): 匹配文档的文本内容
    '''
    results = collection.query(
        query_texts=query,
        n_results=n_results,
    )
    
    return results


@router.post("/add-docs")
async def add_docs_to_collection(
    name: str,
    documents: List[Document],
    embedding_model: LC_Embeddings = Depends(create_lc_embedding_model),
    knowledgebase_collections: List[str] = Depends(list_chroma_collections)
) -> None:
    '''
    将文档添加到指定名称的 collection 中，并进行向量化。

    Args:
        name (str): 知识库名称
        documents (List[str]): 文档的具体内容列表
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
    '''
    # Check if collection exists
    if name not in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection does not exist")
    
    collection_in_client = lcchroma.Chroma(
        collection_name=name,
        persist_directory=KNOWLEDGE_BASE_PATH,
        embedding_function=embedding_model
    )

    # Add documents to collection
    collection_in_client.add_documents(
        documents=documents
    )


@router.post("/delete-whole-file-in-collection")
async def delete_whole_file_in_collection(
    name: str,
    files_name: str,
    knowledgebase_collections: List[str] = Depends(list_chroma_collections),
    embedding_model: LC_Embeddings = Depends(create_lc_embedding_model)
) -> None:
    '''
    删除指定 collection 中的指定文件。

    Args:
        name (str): 知识库名称
        file_name (str): 待删除的文件名
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
    '''
    # Check if collection exists
    if name not in knowledgebase_collections:
        raise HTTPException(status_code=400, detail="Collection does not exist")
    
    # Delete file from collection
    # Initialize an empty list to store the ids
    ids_for_target_file = []

    # Get the documents from the collection
    client = lcchroma.Chroma(
        collection_name=name,
        persist_directory=KNOWLEDGE_BASE_PATH,
        embedding_function=embedding_model
    )

    metadata = client.get()
    
    # Loop over the metadata
    for i in range(len(metadata['metadatas'])):
        # Check if the source matches the target file
        # We only compare the last part of the path (the actual file name)
        if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == files_name:
            # If it matches, add the corresponding id to the list
            ids_for_target_file.append(metadata['ids'][i])

    client.delete(ids=ids_for_target_file)

    
@router.post("/delete-specific-splitted-document")
async def delete_specific_chunk_document(
    chunk_document_content: str,
    collection: chromadb.Collection = Depends(get_chroma_specific_collection)
) -> None:
    '''
    删除指定 collection 中的分割后的文件块。
    
    Args:
        name (str): 知识库名称
        chunk_document_name (str): 待删除的文件块名
        knowledgebase_collections (List[str], optional): 知识库的所有 collection
        embedding_model (chromadb.EmbeddingFunction, optional): 嵌入模型
        
    '''
    collection.delete(where_document={"$contains": chunk_document_content})
