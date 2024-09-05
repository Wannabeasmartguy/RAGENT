from fastapi import APIRouter, Depends
from fastapi import HTTPException
from pydantic import BaseModel

import openai

from autogen.oai import OpenAIWrapper
from autogen import ConversableAgent

from typing import List, Dict, Literal, Any

from api.routers.chat import LLMConfig, LLMParams
from api.routers.knowledgebase import EmbeddingModelConfig
from llm.aoai.tools.tools import TO_TOOLS
from utils.basic_utils import dict_filter

router = APIRouter(
    prefix="/agentchat",
    tags=["agentchat"],
    responses={404: {"description": "Not found"}},
)


@router.post("/autogen/create-function-call-agent-response")
async def create_function_call_agent_response(
    message: str,
    llm_config: LLMConfig,
    llm_params: LLMParams | None,
    tools: List[str] = [],
) :
    '''
    创建一个agentchat的function call响应
    '''
    config = llm_config.dict()
    params = llm_params.dict()

    try:
        if "stream" in params:
            del params["stream"]
    except:
        pass

    config.update(params)

    all_tools = TO_TOOLS
    selected_tools = dict_filter(all_tools, tools)

    assistant = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant. "
        "You can help with web scraper. "
        "Return 'TERMINATE' when the task is done.",
        llm_config={
            "config_list": [
               config 
            ],
        }
    )

    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )

    for tool_name in selected_tools:
        tool = selected_tools[tool_name]
        # Register the tool signature with the assistant agent.
        assistant.register_for_llm(name=tool["name"], description=tool["description"])(tool["func"])

        # Register the tool function with the user proxy agent.
        user_proxy.register_for_execution(name=tool["name"])(tool["func"])

    result = user_proxy.initiate_chat(
        assistant,
        message=message,
        max_turns=10
    )
    
    return result.chat_history