import streamlit as st

from typing import Dict, List

from configs.basic_config import I18nAuto, SUPPORTED_LANGUAGES
from configs.pydantic_model.message import CozeBotResponse, Message


i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])


def display_cozebot_response(coze_response: CozeBotResponse | Dict[str, str]) -> CozeBotResponse:
    """
    Display the cozebot response in a Streamlit app.

    Args:
        coze_response (CozeBotResponse | Dict[str, str]): The cozebot response to display.
        
    Returns:
        CozeBotResponse: The cozebot response with the messages displayed in the app.
    """
    follow_up_detect = 0

    if isinstance(coze_response, dict):
        coze_response = CozeBotResponse(**coze_response)

    messages_list = []

    for message in coze_response.messages:
        # 去除掉tool_response和verbose类型的消息
        if message.content_type != 'card' and message.type not in ('tool_response', 'verbose'):
            # 根据类型的不同，采用不同的方式展示消息
            if message.type == 'answer':
                st.write(message.content)
            elif message.type == 'function_call':
                function_call = st.popover(i18n('Function Call'))
                function_call.write(message.content)
            elif message.type == 'follow_up':
                follow_up_detect += 1
                if follow_up_detect == 1:
                    st.write(i18n('---'))
                    st.write(i18n('Follow Up'))
                st.button(
                    label=message.content
                )
        
        # 构建返回值
        if message.type in ('answer', 'function_call', 'tool_response'):
            processed_message = Message.model_validate({
                'role': message.role,
                'type': message.type,
                'content': message.content,
                'content_type': message.content_type
            })
            messages_list.append(processed_message)
    
    coze_response.messages = messages_list
    return coze_response


def display_coze_conversation(conversation: List[Dict[str, str]]) -> None:
    """
    Display the cozebot conversation in a Streamlit app.
    
    Args:
        conversation (List[Dict[str, str]]): The cozebot conversation to display.
        
    Returns:
        None
    """
    for message in conversation:
        if message['role'] == 'user':
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        elif message['role'] == 'assistant':
            with st.chat_message(message["role"]):
                display_cozebot_response(message['content'])