import streamlit as st
import os
from configs.basic_config import I18nAuto, set_pages_configs_in_common
from llm.Agent.pre_built import reflection_agent_with_nested_chat
from llm.aoai.completion import aoai_config_generator
from llm.groq.completion import groq_config_generator

from autogen.cache import Cache

i18n = I18nAuto()

# Initialize chat history, to avoid error when reloading the page
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

VERSION = "0.0.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
set_pages_configs_in_common(version=VERSION,title="RAGenT-AgentChat",page_icon_path=logo_path)


def write_chat_history(chat_history):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


st.write("目前，RAGenT的两种对话模式是共用session.state的；并且AgentChat是不能携带对话记忆的，边栏的“历史消息对话数”控制的是Agent自行对话时的对话记忆长度。")
write_chat_history(st.session_state.chat_history)


def model_selector(model_type):
    if model_type == "OpenAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Groq":
        return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    else:
        return None
    

with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.write(i18n("Sub pages"))
    st.page_link("pages/AgentChat_Setting.py", label=i18n("⚙️ AgentChat Setting"))
    st.page_link("pages/2_📖Knowledge_Base_Setting.py", label=(i18n("📖 Knowledge Base Setting")))
    st.write('---')

    agent_type = st.selectbox(
        label=i18n("Agent type"),
        options=["Reflection","RAG"],
        key="agent_type"
    )

    select_box0 = st.selectbox(
        label=i18n("Model type"),
        options=["OpenAI","Groq"],
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
        value=32,
        step=1,
        key="history_length"
    )

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


# 根据选择的模型和类型，生成相应的 config_list
if st.session_state["model_type"] == "OpenAI":
    config_list = aoai_config_generator(model=st.session_state["model"])
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(model=st.session_state["model"])

if agent_type == "Reflection":
    user_proxy, writing_assistant, reflection_assistant = reflection_agent_with_nested_chat(config_list=config_list,max_message=history_length)

# st.write(type(user_proxy))

if prompt := st.chat_input("What is up?"):
    if agent_type == "Reflection":
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Use Cache.disk to cache the generated responses.
        # This is useful when the same request to the LLM is made multiple times.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                with Cache.disk(cache_seed=42) as cache:
                    result = user_proxy.initiate_chat(
                        writing_assistant,
                        message=prompt,
                        max_turns=2,
                        cache=cache,
                    )
            # result 是一个 list[dict]，遍历取出并write出来
            result_chat_his = result.chat_history
            st.session_state.chat_history.extend(result_chat_his[1:])

            with st.container(height=500,border=True):
                for message in result_chat_his[1:]:
                    st.markdown(message["content"])
                    # Add assistant message to chat history
                    # st.session_state.chat_history.append({"role": "assistant", "content": message["content"]})
                    st.write("---")
                # Add assistant message to chat history
            st.session_state.summary = result.summary
            st.write("# 概要")
            st.markdown(result.summary)

# st.write(st.session_state.chat_history)
        