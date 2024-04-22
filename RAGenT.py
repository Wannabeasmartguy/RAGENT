import streamlit as st
import autogen
import numpy as np
from PIL import Image

from autogen.oai.openai_utils import config_list_from_dotenv
from autogen.agentchat.contrib.capabilities import transforms

import os
from dotenv import load_dotenv
load_dotenv()

from llm.aoai.completion import AzureOpenAICompletionClient
from llm.ollama.completion import OllamaCompletionClient,get_ollama_model_list
from llm.groq.completion import GroqCompletionClient,groq_config_generator
from configs.basic_config import I18nAuto
from copy import deepcopy


# i18n = I18nAuto(language="en-US")
i18n = I18nAuto()

VERSION = "0.0.1"
st.set_page_config(
    page_title="RAGenT",
    page_icon=os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png"),
    initial_sidebar_state="expanded",
    menu_items={
            'Get Help': 'https://github.com/Wannabeasmartguy/RAGenT',
            'Report a bug': "https://github.com/Wannabeasmartguy/RAGenT/issues",
            'About': f"""欢迎使用 RAGenT WebUI {VERSION}！"""
        }    
)


st.title("RAGenT")
 

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
def write_chat_history(chat_history):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

write_chat_history(st.session_state.chat_history)
            

def model_selector(model_type):
    if model_type == "OpenAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        try:
           model_list = get_ollama_model_list() 
           return model_list
        except:
            return ["qwen:7b-chat"]
    elif model_type == "Groq":
        return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    else:
        return None


with st.sidebar:
    logo_path = os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png")
    image = Image.open(logo_path)
    image_array = np.array(image)
    st.image(logo_path)

    chat_type = st.selectbox(
        label=i18n("Chat type"),
        options=["LLM Chat","Agent Chat"],
        key="chat_type"
    )
    select_box0 = st.selectbox(
        label=i18n("Model type"),
        options=["OpenAI","Ollama","Groq"],
        key="model_type",
        # on_change=lambda: model_selector(st.session_state["model_type"])
    )

    select_box1 = st.selectbox(
        label=i18n("Model"),
        options=model_selector(st.session_state["model_type"]),
        key="model"
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

    # 添加一个按键来清空聊天历史
    clear_button = st.button(
        label=i18n("Clear chat history")
    )
    if clear_button:
        st.session_state.chat_history = []
        write_chat_history(st.session_state.chat_history)

    # 添加一个按键来导出易于阅读的聊天历史到本地文件夹中
    export_button = st.button(
        label=i18n("Export chat history")
    )
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
        # st.markdown(chat_history)
        # 将Markdown保存到本地文件夹中
        with open("chat_history.md", "w") as f:
            f.write(chat_history)
        st.success("Chat history exported to chat_history.md")


# load config list from .env file
# config_list = config_list_from_dotenv(
#     dotenv_file_path=".env",
#     model_api_key_map={
#         "gpt-4":{
#             "api_key_env_var":"AZURE_OAI_KEY",
#             "api_type": "API_TYPE",
#             "base_url": "AZURE_OAI_ENDPOINT",
#             "api_version": "API_VERSION",
#         },
#         "gpt-3.5-turbo":{
#             "api_key_env_var":"AZURE_OAI_KEY",
#             "api_type": "API_TYPE",
#             "base_url": "AZURE_OAI_ENDPOINT",
#             "api_version": "API_VERSION",
#         }
#     }
# )

if st.session_state["model_type"] == "OpenAI":
    client = AzureOpenAICompletionClient()
elif st.session_state["model_type"] == "Ollama":
    client = OllamaCompletionClient()
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(
        model = st.session_state["model"]
    )
    client = GroqCompletionClient(
        config=config_list[0]
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
            processed_messages = max_msg_transfrom.apply_transform(deepcopy(st.session_state.chat_history))
            raw_response = client.create_completion(
                model=st.session_state["model"],
                messages=processed_messages
            )
            # 根据 Client 的不同，解析方法有小不同
            if isinstance(client, AzureOpenAICompletionClient):
                response = client.client.extract_text_or_completion_object(raw_response)[0]
            elif isinstance(client, OllamaCompletionClient):
                response = client.extract_text_or_completion_object(raw_response)[0]
            elif isinstance(client, GroqCompletionClient):
                response = client.extract_text_or_completion_object(raw_response)[0]

        st.write(response)
        st.write(f"response cost: ${raw_response.cost}")
    st.session_state.chat_history.append({"role": "assistant", "content": response})

