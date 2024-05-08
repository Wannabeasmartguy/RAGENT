import streamlit as st
import autogen

from autogen.oai.openai_utils import config_list_from_dotenv
from autogen.agentchat.contrib.capabilities import transforms

import os
from dotenv import load_dotenv
load_dotenv()

from llm.aoai.completion import AzureOpenAICompletionClient,aoai_config_generator
from llm.ollama.completion import OllamaCompletionClient,get_ollama_model_list
from llm.groq.completion import GroqCompletionClient,groq_config_generator
from llm.llamafile.completion import LlamafileCompletionClient,llamafile_config_generator
from configs.basic_config import I18nAuto,set_pages_configs_in_common
from copy import deepcopy


# i18n = I18nAuto(language="en-US")
i18n = I18nAuto()

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
    elif model_type == "Llamafile":
        return ["Noneed"]
    else:
        return None


with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    select_box0 = st.selectbox(
        label=i18n("Model type"),
        options=["OpenAI","Ollama","Groq","Llamafile"],
        key="model_type",
        # on_change=lambda: model_selector(st.session_state["model_type"])
    )

    select_box1 = st.selectbox(
        label=i18n("Model"),
        options=model_selector(st.session_state["model_type"]),
        key="model"
    )

    if select_box0 == "Llamafile":
        with st.expander(label=i18n("Llamafile config")):
            llamafile_endpoint = st.text_input(
                label=i18n("Llamafile endpoint"),
                value="http://127.0.0.1:8080/v1",
                key="llamafile_endpoint"
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
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
        # st.markdown(chat_history)
        # 将Markdown保存到本地文件夹中
        with open("chat_history.md", "w") as f:
            f.write(chat_history)
        st.toast(body="Chat history exported to chat_history.md",icon="🎉")

    st.write("---")

    saved_dialog = st.selectbox(
        label=i18n("Saved dialog"),
        options=["None"],
    )

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
    config_list = aoai_config_generator()
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
elif st.session_state["model_type"] == "Llamafile":
    config_list = llamafile_config_generator(base_url=st.session_state["llamafile_endpoint"])
    client = LlamafileCompletionClient(
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
            if not config_list[0].get("params",{}).get("stream",False):
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
                elif isinstance(client, LlamafileCompletionClient):
                    response = client.extract_text_or_completion_object(raw_response)[0]
                else:
                    raise NotImplementedError("Unsupported client type")
            else:
                response = client.create_completion_stream(
                    model=st.session_state["model"],
                    messages=processed_messages
                )
                # st.write_stream(response)

        if not config_list[0].get("params",{}).get("stream",False):
            st.write(response)
            st.write(f"response cost: ${raw_response.cost}")
            
    st.session_state.chat_history.append({"role": "assistant", "content": response})

