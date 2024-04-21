import streamlit as st
import autogen

from autogen.oai.openai_utils import config_list_from_dotenv

import os
from dotenv import load_dotenv
load_dotenv()

from aoai.completion import AzureOpenAICompletionClient
from ollama.completion import OllamaCompletionClient,get_ollama_model_list
from copy import deepcopy


st.title("RAGenT")
 
# style = """
# <style>
# .memo-box {
#     border: 1px solid #ccc;
#     padding: 10px;
#     margin-bottom: 20px;
# }
# .tag {
#     font-size: 12px;
#     color: #888;
# }
# </style>
# """
# html = """
# <div class="memo-box">
#     <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
#     <span class="tag">#tag</span>
# </div>
# """
# st.components.v1.html(style + html)

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
    else:
        return None



select_box0 = st.sidebar.selectbox(
    label="Model type",
    options=["OpenAI","Ollama"],
    key="model_type",
    # on_change=lambda: model_selector(st.session_state["model_type"])
)

select_box1 = st.sidebar.selectbox(
    label="Model",
    options=model_selector(st.session_state["model_type"]),
    key="model"
)

if_agent_mode = st.sidebar.checkbox("Agent Mode", key="if_agent_mode")
if if_agent_mode:
    st.sidebar.write("Agent Mode is on")

# 添加一个按键来清空聊天历史

clear_button = st.sidebar.button("清空聊天记录")
if clear_button:
    st.session_state.chat_history = []
    write_chat_history(st.session_state.chat_history)

# 添加一个按键来导出易于阅读的聊天历史到本地文件夹中
export_button = st.sidebar.button("导出聊天记录")
if export_button:
    # 将聊天历史导出为Markdown
    chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
    # st.markdown(chat_history)
    # 将Markdown保存到本地文件夹中
    with open("chat_history.md", "w") as f:
        f.write(chat_history)
    st.success("Chat history exported to chat_history.md")


# Set OpenAI API key from Streamlit secrets
# client = AzureOpenAI(
#   azure_endpoint = os.getenv('AZURE_OAI_ENDPOINT'), 
#   api_key = os.getenv('AZURE_OAI_KEY'),  
#   api_version = os.getenv('API_VERSION')
# )


# load config list from .env file
config_list = config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={
        "gpt-4":{
            "api_key_env_var":"AZURE_OAI_KEY",
            "api_type": "API_TYPE",
            "base_url": "AZURE_OAI_ENDPOINT",
            "api_version": "API_VERSION",
        },
        "gpt-3.5-turbo":{
            "api_key_env_var":"AZURE_OAI_KEY",
            "api_type": "API_TYPE",
            "base_url": "AZURE_OAI_ENDPOINT",
            "api_version": "API_VERSION",
        }
    }
)

if st.session_state["model_type"] == "OpenAI":
    client = AzureOpenAICompletionClient()
elif st.session_state["model_type"] == "Ollama":
    client = OllamaCompletionClient()

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
            raw_response = client.create_completion(
                model=st.session_state["model"],
                chat_history=st.session_state.chat_history
            )
            # 根据 Client 的不同，解析方法有小不同
            if isinstance(client, AzureOpenAICompletionClient):
                response = client.client.extract_text_or_completion_object(raw_response)[0]
            elif isinstance(client, OllamaCompletionClient):
                response = client.extract_text_or_completion_object(raw_response)[0]

        st.write(response)
        st.write(f"response cost: ${raw_response.cost}")
    st.session_state.chat_history.append({"role": "assistant", "content": response})

