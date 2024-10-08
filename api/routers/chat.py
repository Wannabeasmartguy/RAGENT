from fastapi import APIRouter, Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import StringIO

from openai import OpenAI

from autogen.oai import OpenAIWrapper

from typing import List, Dict, Literal, Optional

from ..dependency import return_supported_sources


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)


class LLMConfig(BaseModel):
    model: str
    base_url: str
    api_key: str
    api_type: Optional[str | None] = None
    api_version: Optional[str | None] = None
    
    class Config:
        extra = "ignore"

class LLMParams(BaseModel):
    temperature: float | None
    top_p: float | None
    max_tokens: int | None = 4096
    stream: bool | None = False

    class Config:
        extra = "ignore"


@router.get("/openai-like-chat/supported-sources")
async def get_supported_sources(support_sources: dict = Depends(return_supported_sources)):
    '''获取支持的 LLM 源'''
    return support_sources
    

@router.post("/openai-like-chat/{source}",deprecated=True)
async def create_completion(
    source: str, 
    llm_config: LLMConfig,
    llm_params: LLMParams | None ,
    messages: List[dict],
    support_sources: dict = Depends(return_supported_sources),
) -> Dict:
    '''
    创建一个 LLM 模型，并使用该模型生成一个响应。
    
    Args:
        source (str):  支持的 LLM 推理源，保存于 dependence.py 中。
        llm_config (LLMConfig): LLM 模型的配置信息。
        llm_params (LLMParams, optional): LLM 模型的参数信息，包括 temperature、top_p 和 max_tokens。
        messages (List[dict]): 完整的对话消息列表。
        support_sources (dict): 支持的 LLM 源列表。
        
    Returns:
        Dict: 生成的响应。
    '''
    if source not in support_sources["sources"]:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # 根据 source 选择不同的处理逻辑
    # 如果 Source 在 sources 中为 "sdk"，则使用 OpenAI SDK 进行处理
    # 如果 Source 在 sources 中为 "request"，则使用 Request 进行处理

    if support_sources["sources"][source] == "sdk":
        client = OpenAIWrapper(
            **llm_config.dict(exclude_unset=True),
            # 禁用缓存
            cache = None,
            cache_seed = None
        )

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
                model = llm_config.model,
            )

        return response
    
    # elif support_sources["sources"][source] == "request":


@router.post("/openai-like-chat/{source}/stream",deprecated=True)
async def create_completion_stream(
    source: str,
    llm_config: LLMConfig,
    llm_params: LLMParams | None ,
    messages: List[dict],
    support_sources: dict = Depends(return_supported_sources),
) -> StreamingResponse:
    '''
    创建一个 LLM 模型，并使用该模型生成一个流式响应。

    Args:
        source (str):  支持的 LLM 推理源，保存于 dependence.py 中。
        llm_config (LLMConfig): LLM 模型的配置信息。
        llm_params (LLMParams, optional): LLM 模型的参数信息，包括 temperature、top_p 和 max_tokens。
        messages (List[dict]): 完整的对话消息列表。
        support_sources (dict): 支持的 LLM 源列表。
    
    Returns:
        StreamingResponse: 生成的流式响应。
    '''
    if source not in support_sources["sources"]:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # 判断 llm_param 中是否 stream == True
    if llm_params.stream:
    
        # 根据 source 选择不同的处理逻辑
        # 如果 Source 在 sources 中为 "sdk"，则使用 OpenAI SDK 进行处理
        # 如果 Source 在 sources 中为 "request"，则使用 Request 进行处理
        
        if support_sources["sources"][source] == "sdk":
            client = OpenAI(
                # **llm_config.dict(exclude_unset=True),
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
            )

            response = client.chat.completions.create(
                messages = messages,
                model = llm_config.model,
                temperature = llm_params.temperature,
                top_p = llm_params.top_p,
                max_tokens = llm_params.max_tokens,
                stream = True
            )

            def streamer():
                for chunk in response:
                    yield chunk.choices[0].delta.content
            
            return StreamingResponse(StringIO(''.join(streamer())), media_type="text/plain")
        
        # elif support_sources["sources"][source] == "request":

    else:
        raise HTTPException(status_code=404, detail="Source stream output unsupported")