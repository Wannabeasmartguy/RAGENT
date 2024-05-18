import streamlit as st
import autogen
import asyncio

from autogen.oai.openai_utils import config_list_from_dotenv
from autogen.agentchat.contrib.capabilities import transforms

import os
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

from api.dependency import APIRequestHandler,SUPPORTED_SOURCES

from llm.aoai.completion import AzureOpenAICompletionClient,aoai_config_generator
from llm.ollama.completion import OllamaCompletionClient,get_ollama_model_list
from llm.groq.completion import GroqCompletionClient,groq_config_generator
from llm.llamafile.completion import LlamafileCompletionClient,llamafile_config_generator
from configs.basic_config import I18nAuto,set_pages_configs_in_common,SUPPORTED_LANGUAGES
from configs.chat_config import ChatProcessor
from utils.basic_utils import model_selector,save_basic_chat_history


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

requesthandler = APIRequestHandler("localhost", 8000)


VERSION = "0.0.1"
logo_path = os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png")
set_pages_configs_in_common(
    version=VERSION,
    title="RAGenT",
    page_icon_path=logo_path
)

st.title("RAGenT")
 

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
@st.cache_data
def write_chat_history(chat_history):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

write_chat_history(st.session_state.chat_history)


with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    select_box0 = st.selectbox(
        label=i18n("Model type"),
        options=["AOAI","OpenAI","Ollama","Groq","Llamafile"],
        key="model_type",
        # on_change=lambda: model_selector(st.session_state["model_type"])
    )

    if select_box0 != "Llamafile":
        select_box1 = st.selectbox(
            label=i18n("Model"),
            options=model_selector(st.session_state["model_type"]),
            key="model"
        )
    elif select_box0 == "Llamafile":
        select_box1 = st.text_input(
            label=i18n("Model"),
            value="Noneed",
            key="model",
            placeholder=i18n("Fill in custom model name. (Optional)")
        )
        with st.popover(label=i18n("Llamafile config"),use_container_width=True):
            llamafile_endpoint = st.text_input(
                label=i18n("Llamafile endpoint"),
                value="http://127.0.0.1:8080/v1",
                key="llamafile_endpoint"
            )
            llamafile_api_key = st.text_input(
                label=i18n("Llamafile API key"),
                value="noneed",
                key="llamafile_api_key",
                placeholder=i18n("Fill in your API key. (Optional)")
            )

    history_length = st.number_input(
        label=i18n("History length"),
        min_value=1,
        value=16,
        step=1,
        key="history_length"
    )
    # 根据历史对话消息数，创建 MessageHistoryLimiter 
    max_msg_transfrom = transforms.MessageHistoryLimiter(max_messages=history_length)

    cols = st.columns(2)
    export_button = cols[0].button(label=i18n("Export chat history"))
    clear_button = cols[1].button(label=i18n("Clear chat history"))
    if clear_button:
        st.session_state.chat_history = []
        write_chat_history(st.session_state.chat_history)
        st.rerun()
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
        # st.markdown(chat_history)
        # 将Markdown保存到本地文件夹中
        with open("chat_history.md", "w") as f:
            f.write(chat_history)
        st.toast(body="Chat history exported to chat_history.md",icon="🎉")

    st.write("---")


    dialog_settings = st.popover(
        label=i18n("Saved dialog settings"),
        use_container_width=True,
        # TODO:未完成保存、删除和读取功能，先disable
        disabled=True,
    )
    
    # 管理已有对话
    saved_dialog = dialog_settings.selectbox(
        label=i18n("Saved dialog"),
        # TODO: 读取本地文件夹中的对话
        options=["None"],
    )
    load_dialog_button = dialog_settings.button(
        label=i18n("Load selected dialog"),
        use_container_width=True,
    )
    delete_dialog_button = dialog_settings.button(
        label=i18n("Delete selected dialog"),
        use_container_width=True,
    )
    if load_dialog_button:
        # TODO: 加载对话
        pass
    if delete_dialog_button:
        # TODO: 删除对话
        pass

    # 保存对话
    dialog_name = dialog_settings.text_input(
        label=i18n("Dialog name"),
    )
    save_dialog_button = dialog_settings.button(
        label=i18n("Save dialog"),
        use_container_width=True,
    )
    if save_dialog_button:
        # TODO: 保存对话到本地文件
        pass


if st.session_state["model_type"] == "OpenAI":
    pass
if st.session_state["model_type"] == "AOAI":
    config_list = aoai_config_generator()
elif st.session_state["model_type"] == "Ollama":
    pass
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(
        model = st.session_state["model"]
    )
elif st.session_state["model_type"] == "Llamafile":
    # 避免因为API_KEY为空字符串导致的请求错误（500）
    if st.session_state["llamafile_api_key"] == "":
        custom_api_key = "noneed"
    else:
        custom_api_key = st.session_state["llamafile_api_key"]
    config_list = llamafile_config_generator(
        model = st.session_state["model"],
        base_url = st.session_state["llamafile_endpoint"],
        api_key = custom_api_key,
    )


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 对消息的数量进行限制
            processed_messages = max_msg_transfrom.apply_transform(deepcopy(st.session_state.chat_history))

            # 非流式调用
            if not config_list[0].get("params",{}).get("stream",False):

                # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中才响应
                # 一般都是在的
                chatprocessor = ChatProcessor(
                    requesthandler=requesthandler,
                    model_type=st.session_state["model_type"],
                    llm_config=config_list[0],
                )

                response = chatprocessor.create_completion(
                    messages=processed_messages
                )
                
            # TODO：流式调用
            else:
                # response = client.create_completion_stream(
                #     model=st.session_state["model"],
                #     messages=processed_messages
                # )
                pass
                # st.write_stream(response)

        if not config_list[0].get("params",{}).get("stream",False):
            if "error" not in response:
                # st.write(response)
                response_content = response["choices"][0]["message"]["content"]
                st.write(response_content)
                cost = response["cost"]
                st.write(f"response cost: ${cost}")

                st.session_state.chat_history.append({"role": "assistant", "content": response_content})    
            else:
                st.error(response)
