from fastapi import APIRouter, Depends
from fastapi import HTTPException
from pydantic import BaseModel

import openai

from autogen.oai import OpenAIWrapper

from typing import List, Dict

from ..dependence import return_supported_sources


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    api_type: str | None
    api_version: str | None

class LLMParams(BaseModel):
    temperature: float | None
    top_p: float | None
    max_tokens: int | None = 4096


@router.get("/openai_like_chat/supported_sources")
async def get_supported_sources(support_sources: dict = Depends(return_supported_sources)):
    return support_sources
    

@router.post("/openai_like_chat/{source}")
async def create_completion(
    source: str, 
    llm_config: LLMConfig,
    llm_params: LLMParams | None ,
    messages: List[dict],
    support_sources: dict = Depends(return_supported_sources),
) -> Dict:
    if source not in support_sources["sources"]:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # 根据 source 选择不同的处理逻辑
    # 如果 Source 在 sources 中为 "sdk" 或 "request_oai"，则使用 OpenAI SDK 进行处理
    # 如果 Source 在 sources 中为 "request_raw"，则使用 Request 进行处理

    # return llm_config
    if support_sources["sources"][source] == "sdk" or support_sources["sources"][source] == "request_oai":
        client = OpenAIWrapper(**llm_config.dict())
        if llm_params:
            response = client.create(
                messages = messages,
                model = llm_config.model,
                temperature = llm_params.temperature,
                top_p = llm_params.top_p,
                max_tokens = llm_params.max_tokens
            )
        else:
            response = client.create(
                messages = messages,
                model=llm_config.model,
            )
        return response
    
    # elif support_sources["sources"][source] == "request_raw":
