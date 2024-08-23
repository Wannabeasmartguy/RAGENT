import streamlit as st
import os
from uuid import uuid4
from loguru import logger

from configs.basic_config import I18nAuto, set_pages_configs_in_common, SUPPORTED_LANGUAGES
from llm.aoai.completion import aoai_config_generator
from llm.groq.completion import groq_openai_config_generator
from llm.llamafile.completion import llamafile_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.fake.completion import fake_agent_chat_completion
from llm.litellm.completion import litellm_config_generator
from utils.basic_utils import (
    model_selector, 
    list_length_transform, 
    oai_model_config_selector, 
    dict_filter
)

from configs.chat_config import AgentChatProcessor, OAILikeConfigProcessor
from configs.knowledge_base_config import ChromaVectorStoreProcessor
from api.dependency import APIRequestHandler
from storage.db.sqlite import SqlAssistantStorage


@st.cache_data
def write_custom_rag_chat_history(chat_history,_sources):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # st.markdown("**Source**:")
                # 展示引用源
                row1 = st.columns(3)
                row2 = st.columns(3)

                for content_source in _sources:
                    if message["response_id"] in content_source:
                        # 获取引用文件
                        response_sources_list = content_source[message["response_id"]]

                for index,pop in enumerate(row1+row2):
                    a = pop.popover(f"引用文件",use_container_width=True)
                    file_name = response_sources_list["metadatas"][index]["source"]
                    file_content = response_sources_list["page_content"][index]
                    a.text(f"引用文件{file_name}")
                    a.code(file_content,language="plaintext")


requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT",8000))

oailike_config_processor = OAILikeConfigProcessor()

vectorstore_processor = ChromaVectorStoreProcessor(
    # 仅需要展示所有的 Collection 即可，故所有参数都为空
    embedding_model_name_or_path="",
    embedding_model_type="huggingface",
)

chat_history_db_dir = os.path.join(os.path.dirname(__file__), "databases", "chat_history")
chat_history_db_file = os.path.join(chat_history_db_dir, "chat_history.db")
if not os.path.exists(chat_history_db_dir):
    os.makedirs(chat_history_db_dir)
chat_history_storage = SqlAssistantStorage(
    table_name="custom_rag_chat_history",
    db_file = chat_history_db_file,
)
if not chat_history_storage.table_exists():
    chat_history_storage.create()


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])


# Initialize RAG chat history, to avoid error when reloading the page
if "custom_rag_chat_history" not in st.session_state:
    st.session_state.custom_rag_chat_history = []
if "custom_rag_sources" not in st.session_state:
    st.session_state.custom_rag_sources = []

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed":{
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "noneed"
        }
    }

if "rag_run_id" not in st.session_state:
    st.session_state.rag_run_id = str(uuid4())

rag_run_id_list = chat_history_storage.get_all_run_ids()
if st.session_state.rag_run_id not in rag_run_id_list:
    st.session_state.rag_current_run_id_index = 0

VERSION = "0.1.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
set_pages_configs_in_common(version=VERSION,title="RAG Chat",page_icon_path=logo_path)


with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")
    st.write(i18n("Sub pages"))
    st.page_link("pages/2_📖Knowledge_Base_Setting.py", label=(i18n("📖 Knowledge Base Setting")))
    st.write('---')

    rag_dialog_settings_tab, rag_model_settings_tab, rag_knowledge_base_settings_tab= st.tabs([i18n("Dialog Settings"), i18n("Model Settings"), i18n("Knowledge Base Settings")])

    with rag_dialog_settings_tab:
        history_length = st.number_input(
            label=i18n("History length"),
            min_value=1,
            value=32,
            step=1,
            key="history_length"
        )

    with rag_model_settings_tab:
        model_choosing_container = st.expander(label=i18n("Model Choosing"),expanded=True)
        select_box0 = model_choosing_container.selectbox(
            label=i18n("Model type"),
            options=["AOAI","OpenAI","Ollama","Groq","Llamafile"],
            key="model_type",
            # on_change=lambda: model_selector(st.session_state["model_type"])
        )

        if select_box0 != "Llamafile":
            select_box1 = model_choosing_container.selectbox(
                label=i18n("Model"),
                options=model_selector(st.session_state["model_type"]),
                key="model"
            )
        elif select_box0 == "Llamafile":
            select_box1 = model_choosing_container.text_input(
                label=i18n("Model"),
                value=oai_model_config_selector(st.session_state.oai_like_model_config_dict)[0],
                key="model",
                placeholder=i18n("Fill in custom model name. (Optional)")
            )
            with model_choosing_container.popover(label=i18n("Llamafile config"),use_container_width=True):
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
                    st.session_state.rag_current_run_id_index = rag_run_id_list.index(st.session_state.rag_run_id)
                    st.rerun()

                delete_oai_like_config_button = st.button(
                    label=i18n("Delete model config"),
                    use_container_width=True,
                    on_click=oailike_config_processor.delete_model_config,
                    args=(oai_like_config_list,)
                )

        reset_model_button = model_choosing_container.button(
            label=i18n("Reset model info"),
            on_click=lambda x: x.cache_clear(),
            args=(model_selector,),
            use_container_width=True
        )

        with st.expander(label=i18n("Model config"),expanded=True):
            max_tokens = st.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=1900,
                step=1,
                key="max_tokens",
                help=i18n("Maximum number of tokens to generate in the completion.Different models may have different constraints, e.g., the Qwen series of models require a range of [0,2000).")
            )
            temperature = st.slider(
                label=i18n("Temperature"),
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                key="temperature",
                help=i18n("'temperature' controls the randomness of the model. Lower values make the model more deterministic and conservative, while higher values make it more creative and diverse. The default value is 0.5.")
            )
            top_p = st.slider(
                label=i18n("Top p"),
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key="top_p",
                help=i18n("Similar to 'temperature', but don't change it at the same time as temperature")
            )
            if_stream = st.toggle(
                label=i18n("Stream"),
                value=True,
                key="if_stream",
                help=i18n("Whether to stream the response as it is generated, or to wait until the entire response is generated before returning it. Default is False, which means to wait until the entire response is generated before returning it.")
            )
            if_tools_call = st.toggle(
                label=i18n("Tools call"),
                value=False,
                key="if_tools_call",
                help=i18n("Whether to enable the use of tools. Only available for some models. For unsupported models, normal chat mode will be used by default."),
                on_change=lambda: logger.info(f"Tools call toggled, current status: {str(st.session_state.if_tools_call)}")
            )

    with rag_knowledge_base_settings_tab:
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
            hybrid_retrieve_weight_placeholder = st.empty()
            if is_hybrid_retrieve:
                hybrid_retrieve_weight = hybrid_retrieve_weight_placeholder.slider(
                    label=i18n("Hybrid retrieve weight"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="hybrid_retrieve_weight"
                )
            else:
                # Prevent error when the checkbox is unchecked
                hybrid_retrieve_weight = 0.0


    export_button_col, clear_button_col = st.columns(2)
    export_button = export_button_col.button(label=i18n("Export chat history"),use_container_width=True)
    clear_button = clear_button_col.button(label=i18n("Clear chat history"),use_container_width=True)
    if clear_button:
        st.session_state.custom_rag_chat_history = []
        st.session_state.custom_rag_sources = []

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
    config_list = groq_openai_config_generator(model=st.session_state["model"])
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
elif st.session_state["model_type"] == "LiteLLM":
    config_list = litellm_config_generator(model=st.session_state["model"])
# logger.debug(f"Config List: {config_list}")


agentchat_processor = AgentChatProcessor(
    requesthandler=requesthandler,
    model_type=select_box0,
    llm_config=config_list[0],
)

write_custom_rag_chat_history(st.session_state.custom_rag_chat_history,st.session_state.custom_rag_sources)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.custom_rag_chat_history.append({"role": "user", "content": prompt})

    processed_messages = list_length_transform(history_length,st.session_state.custom_rag_chat_history)
    # 在 invoke 的 messages 中去除 response_id
    processed_messages = [dict_filter(item, ["role", "content"]) for item in processed_messages]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agentchat_processor.create_custom_rag_response(
                collection_name=collection_selectbox,
                messages=processed_messages,
                is_rerank=is_rerank,
                is_hybrid_retrieve=is_hybrid_retrieve,
                hybrid_retriever_weight=hybrid_retrieve_weight
            )
        
        response = response.model_dump()
        # 将回答添加入 st.sesstion
        st.session_state.custom_rag_chat_history.append({"role": "assistant", "content": response["answer"]["choices"][0]["message"]["content"], "response_id": response["response_id"]})

        # 将引用sources添加到 st.session
        st.session_state.custom_rag_sources.append({response["response_id"]: response["source_documents"]})
        
        # 展示回答
        st.write(response["answer"]["choices"][0]["message"]["content"])

        # 展示引用源
        row1 = st.columns(3)
        row2 = st.columns(3)

        for content_source in st.session_state.custom_rag_sources:
            if response["response_id"] in content_source:
                # 获取引用文件
                response_sources_list = content_source[response["response_id"]]

        for index,pop in enumerate(row1+row2):
            a = pop.popover(f"引用文件",use_container_width=True)
            file_name = response_sources_list["metadatas"][index]["source"]
            file_content = response_sources_list["page_content"][index]
            a.text(f"引用文件{file_name}")
            a.code(file_content,language="plaintext")