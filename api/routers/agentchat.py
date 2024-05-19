from fastapi import APIRouter, Depends
from fastapi import HTTPException
from pydantic import BaseModel

import openai

from autogen.oai import OpenAIWrapper

from typing import List, Dict, Literal, Any

from api.routers.chat import LLMConfig, LLMParams
from api.routers.knowledgebase import EmbeddingModelConfig
from lc.rag.basic import LCOpenAILikeRAGManager


router = APIRouter(
    prefix="/agentchat",
    tags=["agentchat"],
    responses={404: {"description": "Not found"}},
)


@router.post("/lc-rag/create-rag-response")
async def create_agentchat_lc_rag_response(
    name: str,
    messages: List[Dict[str, str]],
    is_rerank: bool,
    is_hybrid_retrieve: bool,
    hybrid_retriever_weight: float,
    llm_config: LLMConfig,
    llm_params: LLMParams
) -> Dict[str, Any]:
    '''
    创建一个agentchat的lc-rag响应

    Args:
        name (str): 知识库名称
        messages (str): 完整的对话上下文
        is_rerank (bool): 是否进行知识库检索结果 rerank
        is_hybrid_retrieve (bool): 是否进行混合检索
        hybrid_retriever_weight (float): 混合检索的权重
        llm_config (LLMConfig): LLM配置
        metadatas (Dict): 知识库查询得到的metadata
    '''
    rag_manager = LCOpenAILikeRAGManager(
        llm_config=llm_config.dict(),
        llm_params=llm_params.dict(),
        collection=name
    )

    if len(messages) == 1:
        prompt = messages[0]["content"]
        chat_history = []
    else:
        prompt = messages[-1]["content"]
        chat_history = messages[:-1]

    response = rag_manager.invoke(
        prompt=prompt,
        chat_history=chat_history,
        is_rerank=is_rerank,
        is_hybrid_retrieve=is_hybrid_retrieve,
        hybrid_retriever_weight=hybrid_retriever_weight,
        sources_num=6
    )

    return response

