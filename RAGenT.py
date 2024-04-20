import streamlit as st
import autogen

from autogen.oai.openai_utils import config_list_from_dotenv

import os
from dotenv import load_dotenv
load_dotenv()

from aoai.completion import CompletionClient


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
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def model_selector(model_type):
    if model_type == "OpenAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        return ["qwen-7b:chat"]
    else:
        return None


with st.container(border=True):
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


    # 添加一个按键来导出易于阅读的聊天历史到本地文件夹中
    export_button = st.sidebar.button("导出聊天记录")
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.messages])
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


client = CompletionClient(model_type=st.session_state["model_type"])

# 使用 config_list 创建 OpenAIWrapper 实例会在选择使用 model 时按照 config_list 中的顺序进行尝试，而非选择 model 指定的模型
# client = OpenAIWrapper(config_list=config_list)


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        raw_response = client.create_completion(
            model=st.session_state["model"],
            chat_history=st.session_state.messages
        )
        response = client.client.extract_text_or_completion_object(raw_response)[0]
        st.write(response)
        st.write(f"response cost: ${raw_response.cost}")
    st.session_state.messages.append({"role": "assistant", "content": response})

