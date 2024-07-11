import streamlit as st
import os
from loguru import logger

from configs.basic_config import I18nAuto, set_pages_configs_in_common, SUPPORTED_LANGUAGES
from llm.Agent.pre_built import reflection_agent_with_nested_chat
from llm.aoai.completion import aoai_config_generator
from llm.groq.completion import groq_config_generator
from llm.llamafile.completion import llamafile_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.fake.completion import fake_agent_chat_completion
from utils.basic_utils import (
    model_selector, 
    split_list_by_key_value, 
    list_length_transform, 
    oai_model_config_selector, 
    reverse_traversal, 
    write_chat_history
)
from llm.aoai.tools.tools import TO_TOOLS

from configs.chat_config import AgentChatProcessor, OAILikeConfigProcessor
from configs.knowledge_base_config import ChromaVectorStoreProcessor
from api.dependency import APIRequestHandler

from autogen.cache import Cache

from typing import List


@st.cache_data
def write_rag_chat_history(chat_history,_sources):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # st.markdown("**Source**:")
                # 展示引用源
                row1 = st.columns(3)
                row2 = st.columns(3)

                for content_source in _sources:
                    if message["content"] in content_source:
                        # 获取引用文件
                        response_sources_list = content_source[message["content"]]

                for index,pop in enumerate(row1+row2):
                    a = pop.popover(f"引用文件",use_container_width=True)
                    file_name = response_sources_list[index].metadata["source"]
                    file_content = response_sources_list[index].page_content
                    a.text(f"引用文件{file_name}")
                    a.code(file_content,language="plaintext")


requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT",8000))

oailike_config_processor = OAILikeConfigProcessor()


vectorstore_processor = ChromaVectorStoreProcessor(
    # 仅需要展示所有的 Collection 即可，故所有参数都为空
    embedding_model_name_or_path="",
    embedding_model_type="huggingface",
)


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

# Initialize chat history, to avoid error when reloading the page
if "agent_chat_history_displayed" not in st.session_state:
    st.session_state.agent_chat_history_displayed = []
if "agent_chat_history_total" not in st.session_state:
    st.session_state.agent_chat_history_total = []

# Initialize RAG chat history, to avoid error when reloading the page
if "rag_chat_history_displayed" not in st.session_state:
    st.session_state.rag_chat_history_displayed = []
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed":{
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "noneed"
        }
    }

# Initialize function call agent chat history, to avoid error when reloading the page
if "function_call_agent_chat_history_displayed" not in st.session_state:
    st.session_state.function_call_agent_chat_history_displayed = []


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


with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")
    st.write(i18n("Sub pages"))
    st.page_link("pages/AgentChat_Setting.py", label=i18n("⚙️ AgentChat Setting"))
    st.page_link("pages/2_📖Knowledge_Base_Setting.py", label=(i18n("📖 Knowledge Base Setting")))
    st.write('---')

    agent_type = st.selectbox(
        label=i18n("Agent type"),
        options=["Reflection","RAG_lc","Function Call"],
        key="agent_type",
        # 显示时删除掉下划线及以后的内容
        format_func=lambda x: x.replace("_lc","")
    )

    if agent_type == "RAG_lc":
        with st.popover(label=i18n("RAG Setting"),use_container_width=True):
            collection_selectbox = st.selectbox(
                label=i18n("Collection"),
                options=vectorstore_processor.list_all_knowledgebase_collections(1)
            )
            is_rerank = st.checkbox(
                label=i18n("Rerank"),
                value=False,
                key="is_rerank"
            )
            is_hybrid_retrieve = st.checkbox(
                label=i18n("Hybrid retrieve"),
                value=False,
                key="is_hybrid_retrieve"
            )
            hybrid_retrieve_weight = st.slider(
                label=i18n("Hybrid retrieve weight"),
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="hybrid_retrieve_weight"
            )
    
    if agent_type == "Function Call":
        with st.expander(label=i18n("Function Call Setting")):
            function_mutiple_selectbox = st.multiselect(
                label=i18n("Functions"),
                options=TO_TOOLS.keys(),
                default=list(TO_TOOLS.keys())[:2],
                help=i18n("Select functions you want to use."),
                # format_func 将所有名称开头的"tool_"去除
                format_func=lambda x: x.replace("tool_","")
            )
            with st.popover(label=i18n("Function Description"),use_container_width=True):
                with st.container(height=300):
                    for tool_name in function_mutiple_selectbox:
                        with st.container(height=150):
                            st.write("#### " + tool_name.replace("tool_",""))
                            st.write(TO_TOOLS[tool_name]["description"])

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
            value=oai_model_config_selector(st.session_state.oai_like_model_config_dict)[0],
            key="model",
            placeholder=i18n("Fill in custom model name. (Optional)")
        )
        with st.popover(label=i18n("Llamafile config"),use_container_width=True):
            llamafile_endpoint = st.text_input(
                label=i18n("Llamafile endpoint"),
                value=oai_model_config_selector(st.session_state.oai_like_model_config_dict)[1],
                key="llamafile_endpoint"
            )
            llamafile_api_key = st.text_input(
                label=i18n("Llamafile API key"),
                value=oai_model_config_selector(st.session_state.oai_like_model_config_dict)[2],
                key="llamafile_api_key",
                placeholder=i18n("Fill in your API key. (Optional)")
            )
            save_oai_like_config_button = st.button(
                label=i18n("Save model config"),
                on_click=oailike_config_processor.update_config,
                args=(select_box1,llamafile_endpoint,llamafile_api_key),
                use_container_width=True
            )
            
            st.write("---")

            oai_like_config_list = st.selectbox(
                label=i18n("Select model config"),
                options=oailike_config_processor.get_config()
            )
            load_oai_like_config_button = st.button(
                label=i18n("Load model config"),
                use_container_width=True,
                type="primary"
            )
            if load_oai_like_config_button:
                st.session_state.oai_like_model_config_dict = oailike_config_processor.get_model_config(oai_like_config_list)
                st.rerun()

            delete_oai_like_config_button = st.button(
                label=i18n("Delete model config"),
                use_container_width=True,
                on_click=oailike_config_processor.delete_model_config,
                args=(oai_like_config_list,)
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
        if agent_type == "RAG_lc":
            st.session_state.rag_chat_history_displayed = []
            st.session_state.rag_sources = []
            write_rag_chat_history(st.session_state.rag_chat_history_displayed,st.session_state.rag_sources)
        elif agent_type == "Reflection":
            st.session_state.agent_chat_history_displayed = []
            st.session_state.agent_chat_history_total = []
            initialize_agent_chat_history(st.session_state.agent_chat_history_displayed,st.session_state.agent_chat_history_total)
        elif agent_type == "Function Call":
            st.session_state.function_call_agent_chat_history_displayed = []
            write_chat_history(st.session_state.function_call_agent_chat_history_displayed)
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
if st.session_state["model_type"] == "AOAI":
    config_list = aoai_config_generator(model=st.session_state["model"])
if st.session_state["model_type"] == "OpenAI":
    config_list = aoai_config_generator(
        model=st.session_state["model"],
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        api_type="openai",
        api_version=None,
    )
if st.session_state["model_type"] == "Ollama":
    config_list = ollama_config_generator(model=st.session_state["model"])
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(model=st.session_state["model"])
elif st.session_state["model_type"] == "Llamafile":
    if st.session_state["llamafile_api_key"] == "":
        custom_api_key = "noneed"
    else:
        custom_api_key = st.session_state["llamafile_api_key"]
    config_list = llamafile_config_generator(
        model = st.session_state["model"],
        base_url = st.session_state["llamafile_endpoint"],
        api_key = custom_api_key,
    )
# logger.debug(f"Config List: {config_list}")


agentchat_processor = AgentChatProcessor(
    requesthandler=requesthandler,
    model_type=select_box0,
    llm_config=config_list[0],
)

if agent_type =="RAG_lc":
    write_rag_chat_history(st.session_state.rag_chat_history_displayed,st.session_state.rag_sources)
elif agent_type == "Reflection":
    # 初始化代理聊天历史
    initialize_agent_chat_history(st.session_state.agent_chat_history_displayed,st.session_state.agent_chat_history_total)
    # 初始化各个 Agent
    user_proxy, writing_assistant, reflection_assistant = reflection_agent_with_nested_chat(config_list=config_list,max_message=history_length)
elif agent_type == "Function Call":
    write_chat_history(st.session_state.function_call_agent_chat_history_displayed)

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


    elif agent_type == "RAG_lc":
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.rag_chat_history_displayed.append({"role": "user", "content": prompt})

        processed_messages = list_length_transform(history_length,st.session_state.rag_chat_history_displayed)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agentchat_processor.create_rag_agent_response_noapi(
                    name=collection_selectbox,
                    messages=processed_messages,
                    is_rerank=is_rerank,
                    is_hybrid_retrieve=is_hybrid_retrieve,
                    hybrid_retriever_weight=hybrid_retrieve_weight
                )
            
            # 将回答添加入 st.sesstion
            st.session_state.rag_chat_history_displayed.append({"role": "assistant", "content": response["answer"]})

            # 将引用sources添加到 st.session
            st.session_state.rag_sources.append({response["answer"]: response["source_documents"]})
            
            # 展示回答
            st.write(response["answer"])

            # 展示引用源
            row1 = st.columns(3)
            row2 = st.columns(3)

            for content_source in st.session_state.rag_sources:
                if response["answer"] in content_source:
                    # 获取引用文件
                    response_sources_list = content_source[response["answer"]]

            for index,pop in enumerate(row1+row2):
                a = pop.popover(f"引用文件",use_container_width=True)
                file_name = response_sources_list[index].metadata["source"]
                file_content = response_sources_list[index].page_content
                a.text(f"引用文件{file_name}")
                a.code(file_content,language="plaintext")
    

    elif agent_type == "Function Call":
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Add user message to chat history
        st.session_state.function_call_agent_chat_history_displayed.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agentchat_processor.create_function_call_agent_response_noapi(
                    message=prompt,
                    tools=function_mutiple_selectbox
                )
                # 返回的是一个完整的chat_history，包含了"None"和""
                # 需要将"None"和""去除
                answer = reverse_traversal(response)
                
                # 将回答添加入 st.sesstion
                st.session_state.function_call_agent_chat_history_displayed.append({"role": "assistant", "content": answer["content"]})

                # 展示回答
                st.write(answer["content"])