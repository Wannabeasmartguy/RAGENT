import streamlit as st
import os
from configs.basic_config import I18nAuto, set_pages_configs_in_common
from llm.Agent.pre_built import reflection_agent_with_nested_chat
from llm.aoai.completion import aoai_config_generator
from llm.groq.completion import groq_config_generator
from llm.llamafile.completion import llamafile_config_generator
from llm.fake.completion import fake_agent_chat_completion
from utils.basic_utils import split_list_by_key_value

from autogen.cache import Cache

from typing import List

# i18n = I18nAuto(language="en-US")
i18n = I18nAuto()

# Initialize chat history, to avoid error when reloading the page
if "agent_chat_history_displayed" not in st.session_state:
    st.session_state.agent_chat_history_displayed = []
if "agent_chat_history_total" not in st.session_state:
    st.session_state.agent_chat_history_total = []

VERSION = "0.0.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
set_pages_configs_in_common(version=VERSION,title="RAGenT-AgentChat",page_icon_path=logo_path)


def annotate_agent_thoughts(thoughts_in_chat_history:List[dict],
                            key:str="if_thought"):
    '''
    This function is used to annotate the agent's thoughts.

    Args:
        thoughts_in_chat_history (List[dict]): A list of dictionaries, each representing a message in the chat history.
        key (str): The key to be used for annotating the agent's thoughts.
    '''
    for index, chat in enumerate(result_chat_his):
        if index == 0 or index == len(result_chat_his) - 1:
            chat[key] = 0
        else:
            chat[key] = 1
    return thoughts_in_chat_history

def display_agent_thoughts(thoughts_in_chat_history:List[dict],
                           key:str="if_thought"):
    '''
    This function is used to display the agent's thoughts.

    Args:
        thoughts_in_chat_history (List[dict]): A list of dictionaries, each representing a message in the chat history.
    '''
    with st.container(border=True):
        # 按顺序展示字典中有"if_thought"字段的内容
        counter = 0
        splitter_counter = 0
        for i,thought in enumerate(thoughts_in_chat_history):
            if thought[key] == 0:
                splitter_counter += 1
                if splitter_counter == 2:
                    break
            if thought[key] == 1:
                with st.expander(f"Agent Thought details({counter+1})"):
                    st.write(thought["content"])
                    counter += 1


def write_agent_chat_history(total_chat_history):
    for message in total_chat_history:
        if message["if_thought"] == 0 and message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if message["if_thought"] == 0 and message["role"] == "assistant":
            with st.chat_message(message["role"]):
                display_agent_thoughts(total_chat_history,key="if_thought")
                st.markdown(message["content"])


def initialize_agent_chat_history(chat_history:List[dict],
                                  chat_history_total:List[dict]):
    round_list = split_list_by_key_value(chat_history_total,key="if_thought",value=0)
    round_counter = 0
    for message in chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_agent_thoughts(round_list[round_counter],key="if_thought")
                round_counter += 1
            st.markdown(message["content"])


initialize_agent_chat_history(st.session_state.agent_chat_history_displayed,st.session_state.agent_chat_history_total)


def model_selector(model_type):
    if model_type == "OpenAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
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
        options=["OpenAI","Groq","Llamafile"],
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
        st.session_state.agent_chat_history_displayed = []
        st.session_state.agent_chat_history_total = []
        initialize_agent_chat_history(st.session_state.agent_chat_history_displayed,st.session_state.agent_chat_history_total)
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.agent_chat_history_total])
        # st.markdown(chat_history)

        # 将Markdown保存到本地文件夹中
        # 如果有同名文件，就为其编号
        filename = "Agent_chat_history.md"
        i = 1
        while os.path.exists(filename):
            filename = f"{i}_{filename}"
            i += 1
            
        with open(filename, "w") as f:
            f.write(chat_history)
        st.toast(body=i18n(f"Chat history exported to {filename}"),icon="🎉")


# 根据选择的模型和类型，生成相应的 config_list
if st.session_state["model_type"] == "OpenAI":
    config_list = aoai_config_generator(model=st.session_state["model"])
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(model=st.session_state["model"])
elif st.session_state["model_type"] == "Llamafile":
    config_list = llamafile_config_generator(model=st.session_state["model"])

if agent_type == "Reflection":
    user_proxy, writing_assistant, reflection_assistant = reflection_agent_with_nested_chat(config_list=config_list,max_message=history_length)

# st.write(type(user_proxy))

if prompt := st.chat_input("What is up?"):
    if agent_type == "Reflection":
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.agent_chat_history_displayed.append({"role": "user", "content": prompt})

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
                # result = fake_agent_chat_completion(prompt)
            # result 是一个 list[dict]，取出并保存
            result_chat_his = result.chat_history
            # 为其中每一个字典添加一个 "if_thought" 字段，用于判断是否是thought
            # 第一个和最后一个为0,其他为1，剩下的内容均完全保留
            annotated_chat_history = annotate_agent_thoughts(result_chat_his)
            st.session_state.agent_chat_history_total.extend(annotated_chat_history) 

            # 展示Agent的详细thought
            display_agent_thoughts(st.session_state.agent_chat_history_total)

            # 仅在 initial_chat 的参数 `summary_method='last_msg'`
            # result.summary 用作对话展示，添加到display中
            st.session_state.agent_chat_history_displayed.append({"role": "assistant", "content": result.summary})
            st.write(result.summary)

# st.write(st.session_state.agent_chat_history_displayed)
        