import streamlit as st
import os
import json
import base64
from typing import Dict, Any
from streamlit_float import *
from uuid import uuid4
from copy import deepcopy
from functools import lru_cache
from loguru import logger
from datetime import datetime
from pydantic import ValidationError

from core.basic_config import (
    I18nAuto,
    set_pages_configs_in_common,
    SUPPORTED_LANGUAGES,
)
from llm.oai.completion import oai_config_generator
from llm.aoai.completion import aoai_config_generator
from llm.groq.completion import groq_openai_config_generator
from llm.llamafile.completion import llamafile_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.litellm.completion import litellm_config_generator
from utils.basic_utils import (
    model_selector,
    list_length_transform,
    oai_model_config_selector,
    dict_filter,
    config_list_postprocess,
    RAG_CHAT_USER_STYLE,
    RAG_CHAT_ASSISTANT_STYLE,
    USER_AVATAR_SVG,
    AI_AVATAR_SVG,
)
from utils.log.logger_config import setup_logger, log_dict_changes
from utils.st_utils import (
    export_dialog,
    back_to_top,
    back_to_bottom,
    float_chat_input_with_audio_recorder,
)

from core.chat_processors import AgentChatProcessor, OAILikeConfigProcessor
from core.kb_processors import (
    ChromaVectorStoreProcessorWithNoApi,
    ChromaCollectionProcessorWithNoApi,
)
from api.dependency import APIRequestHandler
from storage.db.sqlite import SqlAssistantStorage
from model.chat.assistant import AssistantRun
from modules.types.rag import BaseRAGResponse
from model.config.embeddings import (
    EmbeddingConfiguration,
)


@lru_cache(maxsize=1)
def get_agentchat_processor():
    return AgentChatProcessor(
        requesthandler=requesthandler,
        model_type=st.session_state.model_type,
        llm_config=st.session_state.rag_chat_config_list[0],
    )


# 在 create_custom_rag_response 调用之前添加
def refresh_retriever():
    get_agentchat_processor.cache_clear()
    # 可能还需要其他刷新操作，比如重新加载向量数据库等


def save_rag_chat_history():
    """
    Save chat history to database.
    Always update the chat entirely, including chat history and sources.
    """
    chat_history_storage.upsert(
        AssistantRun(
            name="assistant",
            run_name=st.session_state.rag_run_name,
            run_id=st.session_state.rag_run_id,
            llm=st.session_state.rag_chat_config_list[0],
            memory={"chat_history": st.session_state.custom_rag_chat_history},
            task_data={"source_documents": st.session_state.custom_rag_sources},
            assistant_data={
                "model_type": st.session_state.model_type,
            },
        )
    )


def display_rag_sources(response_sources: Dict[str, Any]):
    import itertools

    num_sources = len(response_sources["metadatas"])
    num_columns = min(3, num_sources)
    visible_sources = min(6, num_sources)

    if num_sources == 0:
        st.toast(i18n("No sources found for this response."))
        return

    rows = [st.columns(num_columns) for _ in range((visible_sources + 2) // 3)]

    def create_source_popover(column, index):
        file_name = response_sources["metadatas"][index]["source"]
        file_content = response_sources["page_content"][index]
        with column.popover(
            i18n("Cited Source") + f" {index+1}", use_container_width=True
        ):
            st.text(i18n("Cited Source") + ": " + file_name)
            if response_sources["distances"] is not None:
                distance = response_sources["distances"][index]
                st.text(i18n("Vector Distance") + ": " + str(distance))
            # 如果使用 reranker，则有 relevance_score
            if "relevance_score" in response_sources["metadatas"][index]:
                relevance_score = response_sources["metadatas"][index]["relevance_score"]
                st.text(i18n("Relevance Score by reranker") + ": " + str(relevance_score))
            st.code(file_content, language="plaintext")

    for index, column in enumerate(itertools.chain(*rows)):
        if index < visible_sources:
            create_source_popover(column, index)

    if num_sources > 6:
        with st.expander(i18n("Show more sources"), expanded=False):
            remaining_sources = num_sources - visible_sources
            remaining_columns = min(3, remaining_sources)
            remaining_rows = [
                st.columns(remaining_columns)
                for _ in range((remaining_sources + 2) // 3)
            ]

            for index, column in enumerate(
                itertools.chain(*remaining_rows), start=visible_sources
            ):
                if index < num_sources:
                    create_source_popover(column, index)


@st.cache_data
def write_custom_rag_chat_history(chat_history, _sources):
    # 将SVG编码为base64
    user_avatar = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"
    ai_avatar = f"data:image/svg+xml;base64,{base64.b64encode(AI_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"

    for message in chat_history:
        with st.chat_message(
            message["role"],
            avatar=user_avatar if message["role"] == "user" else ai_avatar,
        ):
            st.html(f"<span class='rag-chat-{message['role']}'></span>")
            st.markdown(message["content"])

            if message["role"] == "assistant":
                rag_sources = _sources[message["response_id"]]
                display_rag_sources(rag_sources)
    combined_style = (
        RAG_CHAT_USER_STYLE.strip() + "\n" + RAG_CHAT_ASSISTANT_STYLE.strip()
    )
    combined_style = combined_style.replace("</style>\n<style>", "")
    st.html(combined_style)


def handle_response(response: BaseRAGResponse, if_stream: bool):
    # 先将引用sources添加到 st.session
    st.session_state.custom_rag_sources.update(
        {response.response_id: response.source_documents}
    )

    if if_stream:
        # 展示回答
        answer = st.write_stream(response.answer)
    else:
        response = response.model_dump()
        answer = response["answer"]["choices"][0]["message"]["content"]
        st.write(answer)

    # 添加回答到 st.session
    response_id = (
        response["response_id"] if isinstance(response, dict) else response.response_id
    )

    st.session_state.custom_rag_chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "response_id": response_id,
        }
    )

    # 保存聊天记录
    save_rag_chat_history()

    # 展示引用源
    response_sources = st.session_state.custom_rag_sources[response_id]
    display_rag_sources(response_sources)


embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")

requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT", 8000))

oailike_config_processor = OAILikeConfigProcessor()

VERSION = "0.1.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, "img", "RAGenT_logo.png")
logo_text = os.path.join(parent_directory, "img", "RAGenT_logo_with_text_horizon.png")
user_avatar = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"
ai_avatar = f"data:image/svg+xml;base64,{base64.b64encode(AI_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"

chat_history_db_dir = os.path.join(parent_directory, "databases", "chat_history")
chat_history_db_file = os.path.join(chat_history_db_dir, "chat_history.db")
if not os.path.exists(chat_history_db_dir):
    os.makedirs(chat_history_db_dir)
chat_history_storage = SqlAssistantStorage(
    table_name="custom_rag_chat_history",
    db_file=chat_history_db_file,
)
if not chat_history_storage.table_exists():
    chat_history_storage.create()


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])


# ********** Initialize session state **********

if "prompt_disabled" not in st.session_state:
    st.session_state.prompt_disabled = False

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed": {"base_url": "http://127.0.0.1:8080/v1", "api_key": "noneed"}
    }

rag_run_id_list = chat_history_storage.get_all_run_ids()
if len(rag_run_id_list) == 0:
    chat_history_storage.upsert(
        AssistantRun(
            name="assistant",
            run_id=str(uuid4()),
            llm=aoai_config_generator(model=model_selector("AOAI")[0], stream=True)[0],
            run_name="New dialog",
            memory={"chat_history": []},
            task_data={"source_documents": {}},
            assistant_data={
                "model_type": "AOAI",
            },
        )
    )
    rag_run_id_list = chat_history_storage.get_all_run_ids()
if "rag_current_run_id_index" not in st.session_state:
    st.session_state.rag_current_run_id_index = 0
while st.session_state.rag_current_run_id_index > len(rag_run_id_list):
    st.session_state.rag_current_run_id_index -= 1
if "rag_run_id" not in st.session_state:
    st.session_state.rag_run_id = rag_run_id_list[
        st.session_state.rag_current_run_id_index
    ]

# initialize config
if "rag_chat_config_list" not in st.session_state:
    st.session_state.rag_chat_config_list = [
        chat_history_storage.get_specific_run(st.session_state.rag_run_id).llm
    ]

# Initialize RAG chat history, to avoid error when reloading the page
if "custom_rag_chat_history" not in st.session_state:
    st.session_state.custom_rag_chat_history = chat_history_storage.get_specific_run(
        st.session_state.rag_run_id
    ).memory["chat_history"]
if "custom_rag_sources" not in st.session_state:
    try:
        st.session_state.custom_rag_sources = chat_history_storage.get_specific_run(
            st.session_state.rag_run_id
        ).task_data["source_documents"]
    except TypeError:
        # TypeError 意味着数据库中没有这个 run_id 的source_documents，因此初始化
        st.session_state.custom_rag_sources = {}

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0


def update_rag_config_in_db_callback():
    """
    Update rag chat llm config in db.
    """
    from copy import deepcopy

    origin_config_list = deepcopy(st.session_state.rag_chat_config_list)
    if st.session_state["model_type"] == "OpenAI":
        config_list = oai_config_generator(
            model=st.session_state.model,
            max_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "AOAI":
        config_list = aoai_config_generator(
            model=st.session_state.model,
            max_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Ollama":
        config_list = ollama_config_generator(
            model=st.session_state.model,
            max_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Groq":
        config_list = groq_openai_config_generator(
            model=st.session_state.model,
            max_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            stream=st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Llamafile":
        try:
            config_list = llamafile_config_generator(
                model=st.session_state.model,
                api_key=st.session_state.llamafile_api_key,
                base_url=st.session_state.llamafile_endpoint,
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                stream=st.session_state.if_stream,
            )
        except (UnboundLocalError, AttributeError) as e:
            # 如果st.session_state没有定义llamafile_api_key，则使用默认值
            logger.warning(f"Error when generating config for llamafile: {e}")
            logger.warning("Just use other existing config")
            # params 可以使用已有配置
            config_list = llamafile_config_generator(
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                stream=st.session_state.if_stream,
            )
    elif st.session_state["model_type"] == "LiteLLM":
        config_list = litellm_config_generator(model=st.session_state["model"])
    st.session_state["rag_chat_config_list"] = config_list
    log_dict_changes(origin_config_list[0], config_list[0])
    chat_history_storage.upsert(
        AssistantRun(
            run_id=st.session_state.rag_run_id,
            llm=config_list[0],
            assistant_data={
                "model_type": st.session_state["model_type"],
                # "system_prompt": st.session_state["system_prompt"],
            },
            updated_at=datetime.now(),
        )
    )

try:
    set_pages_configs_in_common(version=VERSION, title="RAG Chat", page_icon_path=logo_path)
except:
    st.rerun()

with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    # st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")
    st.write(i18n("Sub pages"))
    st.page_link(
        "pages/2_📖Knowledge_Base_Setting.py", label=(i18n("📖 Knowledge Base Setting"))
    )

    rag_dialog_settings_tab, rag_model_settings_tab, rag_knowledge_base_settings_tab = (
        st.tabs(
            [
                i18n("Dialog Settings"),
                i18n("Model Settings"),
                i18n("Knowledge Base Settings"),
            ]
        )
    )

    with rag_dialog_settings_tab:
        st.write(i18n("Dialogues list"))
        dialogs_container = st.container(height=250, border=True)

        def rag_saved_dialog_change_callback():
            origin_config_list = deepcopy(st.session_state.rag_chat_config_list)
            st.session_state.rag_run_id = st.session_state.rag_saved_dialog.run_id
            st.session_state.rag_current_run_id_index = (
                chat_history_storage.get_all_run_ids().index(
                    st.session_state.rag_run_id
                )
            )
            st.session_state.rag_chat_config_list = [
                chat_history_storage.get_specific_run(
                    st.session_state.rag_saved_dialog.run_id
                ).llm
            ]
            logger.info(f"RAG dialog change, selected dialog name: {st.session_state.rag_saved_dialog.run_name}, selected dialog id: {st.session_state.rag_run_id}")
            log_dict_changes(
                original_dict=origin_config_list[0],
                new_dict=st.session_state.rag_chat_config_list[0],
            )
            try:
                st.session_state.custom_rag_chat_history = (
                    chat_history_storage.get_specific_run(
                        st.session_state.rag_saved_dialog.run_id
                    ).memory["chat_history"]
                )
                st.session_state.custom_rag_sources = (
                    chat_history_storage.get_specific_run(
                        st.session_state.rag_saved_dialog.run_id
                    ).task_data["source_documents"]
                )
            except (TypeError, ValidationError):
                st.session_state.custom_rag_chat_history = []
                st.session_state.custom_rag_sources = {}

        saved_dialog = dialogs_container.radio(
            label=i18n("Saved dialog"),
            options=chat_history_storage.get_all_runs(),
            format_func=lambda x: (
                x.run_name[:15] + "..." if len(x.run_name) > 15 else x.run_name
            ),
            index=st.session_state.rag_current_run_id_index,
            label_visibility="collapsed",
            key="rag_saved_dialog",
            on_change=rag_saved_dialog_change_callback,
        )

        add_dialog_column, delete_dialog_column = st.columns([1, 1])
        with add_dialog_column:

            def add_rag_dialog_callback():
                st.session_state.rag_run_id = str(uuid4())
                chat_history_storage.upsert(
                    AssistantRun(
                        name="assistant",
                        run_id=st.session_state.rag_run_id,
                        run_name="New dialog",
                        llm=aoai_config_generator(
                            model=model_selector("AOAI")[0], stream=True
                        )[0],
                        memory={"chat_history": []},
                        task_data={
                            "source_documents": {},
                        },
                    )
                )
                st.session_state.rag_current_run_id_index = 0
                st.session_state.rag_chat_config_list = [
                    chat_history_storage.get_specific_run(
                        st.session_state.rag_run_id
                    ).llm
                ]
                st.session_state.custom_rag_chat_history = []
                st.session_state.custom_rag_sources = {}
                logger.info(f"Add a new RAG dialog, added dialog name: {st.session_state.rag_run_name}, added dialog id: {st.session_state.rag_run_id}")

            add_dialog_button = st.button(
                label=i18n("Add a new dialog"),
                use_container_width=True,
                on_click=add_rag_dialog_callback,
            )
        with delete_dialog_column:

            def delete_rag_dialog_callback():
                chat_history_storage.delete_run(st.session_state.rag_run_id)
                if len(chat_history_storage.get_all_run_ids()) == 0:
                    st.session_state.rag_run_id = str(uuid4())
                    chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_id=st.session_state.rag_run_id,
                            llm=aoai_config_generator(model=model_selector("AOAI")[0])[
                                0
                            ],
                            run_name="New dialog",
                            memory={"chat_history": []},
                            task_data={
                                "source_documents": {},
                            },
                        )
                    )
                    st.session_state.rag_chat_config_list = [
                        chat_history_storage.get_specific_run(
                            st.session_state.rag_run_id
                        ).llm
                    ]
                    st.session_state.custom_rag_chat_history = []
                    st.session_state.custom_rag_sources = {}
                else:
                    st.session_state.rag_run_id = (
                        chat_history_storage.get_all_run_ids()[0]
                    )
                    st.session_state.rag_chat_config_list = [
                        chat_history_storage.get_specific_run(
                            st.session_state.rag_run_id
                        ).llm
                    ]
                    st.session_state.custom_rag_chat_history = (
                        chat_history_storage.get_specific_run(
                            st.session_state.rag_run_id
                        ).memory["chat_history"]
                    )
                    st.session_state.custom_rag_sources = (
                        chat_history_storage.get_specific_run(
                            st.session_state.rag_run_id
                        ).task_data["source_documents"]
                    )
                logger.info(f"Delete a RAG dialog, deleted dialog name: {st.session_state.rag_run_name}, deleted dialog id: {st.session_state.rag_run_id}")

            delete_dialog_button = st.button(
                label=i18n("Delete selected dialog"),
                use_container_width=True,
                on_click=delete_rag_dialog_callback,
            )

        # 保存对话
        def get_run_name():
            try:
                run_name = saved_dialog.run_name
            except:
                run_name = "RAGenT"
            return run_name

        def get_all_runnames():
            runnames = []
            runs = chat_history_storage.get_all_runs()
            for run in runs:
                runnames.append(run.run_name)
            return runnames

        st.write(i18n("Dialogues details"))

        dialog_details_settings_popover = st.expander(
            label=i18n("Dialogues details"),
            # use_container_width=True
        )

        def rag_dialog_name_change_callback():
            origin_run_name = saved_dialog.run_name
            chat_history_storage.upsert(
                AssistantRun(
                    run_name=st.session_state.rag_run_name,
                    run_id=st.session_state.rag_run_id,
                )
            )
            logger.info(f"RAG dialog name changed from {origin_run_name} to {st.session_state.rag_run_name}.(run_id: {st.session_state.rag_run_id})")
            st.session_state.rag_current_run_id_index = rag_run_id_list.index(
                st.session_state.rag_run_id
            )

        dialog_name = dialog_details_settings_popover.text_input(
            label=i18n("Dialog name"),
            value=get_run_name(),
            key="rag_run_name",
            on_change=rag_dialog_name_change_callback,
        )

        # dialog_details_settings_popover.text_area(
        #     label=i18n("System Prompt"),
        #     value=get_system_prompt(saved_dialog.run_id),
        #     height=300,
        #     key="system_prompt",
        # )

        history_length = dialog_details_settings_popover.number_input(
            label=i18n("History length"),
            min_value=1,
            value=32,
            step=1,
            help=i18n(
                "The number of messages to keep in the llm memory."
            ),
            key="history_length",
        )

    with rag_model_settings_tab:
        model_choosing_container = st.expander(
            label=i18n("Model Choosing"), expanded=True
        )

        def get_model_type_index():
            options = ["AOAI", "OpenAI", "Ollama", "Groq", "Llamafile"]
            try:
                return options.index(
                    chat_history_storage.get_specific_run(
                        st.session_state.rag_run_id
                    ).assistant_data["model_type"]
                )
            except:
                return 0

        select_box0 = model_choosing_container.selectbox(
            label=i18n("Model type"),
            options=["AOAI", "OpenAI", "Ollama", "Groq", "Llamafile"],
            index=get_model_type_index(),
            key="model_type",
            on_change=update_rag_config_in_db_callback,
        )

        with st.expander(label=i18n("Model config"), expanded=True):
            max_tokens = st.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=config_list_postprocess(st.session_state.rag_chat_config_list)[
                    0
                ].get("max_tokens", 1900),
                step=1,
                key="max_tokens",
                on_change=update_rag_config_in_db_callback,
                help=i18n(
                    "Maximum number of tokens to generate in the completion.Different models may have different constraints, e.g., the Qwen series of models require a range of [0,2000)."
                ),
            )
            temperature = st.slider(
                label=i18n("Temperature"),
                min_value=0.0,
                max_value=1.0,
                value=config_list_postprocess(st.session_state.rag_chat_config_list)[
                    0
                ].get("temperature", 0.5),
                step=0.1,
                key="temperature",
                on_change=update_rag_config_in_db_callback,
                help=i18n(
                    "'temperature' controls the randomness of the model. Lower values make the model more deterministic and conservative, while higher values make it more creative and diverse. The default value is 0.5."
                ),
            )
            top_p = st.slider(
                label=i18n("Top p"),
                min_value=0.0,
                max_value=1.0,
                value=config_list_postprocess(st.session_state.rag_chat_config_list)[
                    0
                ].get("top_p", 0.5),
                step=0.1,
                key="top_p",
                on_change=update_rag_config_in_db_callback,
                help=i18n(
                    "Similar to 'temperature', but don't change it at the same time as temperature"
                ),
            )
            if_stream = st.toggle(
                label=i18n("Stream"),
                value=config_list_postprocess(st.session_state.rag_chat_config_list)[
                    0
                ].get("stream", True),
                key="if_stream",
                on_change=update_rag_config_in_db_callback,
                help=i18n(
                    "Whether to stream the response as it is generated, or to wait until the entire response is generated before returning it. Default is False, which means to wait until the entire response is generated before returning it."
                ),
            )
            # if_tools_call = st.toggle(
            #     label=i18n("Tools call"),
            #     value=False,
            #     key="if_tools_call",
            #     help=i18n(
            #         "Whether to enable the use of tools. Only available for some models. For unsupported models, normal chat mode will be used by default."
            #     ),
            #     on_change=lambda: logger.info(
            #         f"Tools call toggled, current status: {str(st.session_state.if_tools_call)}"
            #     ),
            # )

        # 为了让 update_config_in_db_callback 能够更新上面的多个参数，需要把model选择放在他们下面
        if select_box0 != "Llamafile":

            def get_selected_non_llamafile_model_index(model_type) -> int:
                try:
                    model = st.session_state.rag_chat_config_list[0].get("model")
                    logger.debug(f"model get: {model}")
                    if model:
                        options = model_selector(model_type)
                        if model in options:
                            options_index = options.index(model)
                            logger.debug(
                                f"model {model} in options, index: {options_index}"
                            )
                            return options_index
                        else:
                            st.session_state.rag_chat_config_list[0].update(
                                {"model": options[0]}
                            )
                            logger.debug(
                                f"model {model} not in options, set model in config list to first option: {options[0]}"
                            )
                            return 0
                except ValueError:
                    logger.warning(
                        f"Model {model} not found in model_selector for {model_type}, returning 0"
                    )
                    return 0

            select_box1 = model_choosing_container.selectbox(
                label=i18n("Model"),
                options=model_selector(st.session_state["model_type"]),
                index=get_selected_non_llamafile_model_index(
                    st.session_state["model_type"]
                ),
                key="model",
                on_change=update_rag_config_in_db_callback,
            )

        elif select_box0 == "Llamafile":

            def get_selected_llamafile_model() -> str:
                if st.session_state.rag_chat_config_list:
                    return st.session_state.rag_chat_config_list[0].get("model")
                else:
                    logger.warning("chat_config_list is empty, using default model")
                    return oai_model_config_selector(
                        st.session_state.oai_like_model_config_dict
                    )[0]

            select_box1 = model_choosing_container.text_input(
                label=i18n("Model"),
                value=get_selected_llamafile_model(),
                key="model",
                placeholder=i18n("Fill in custom model name. (Optional)"),
            )

            with model_choosing_container.popover(
                label=i18n("Llamafile config"), use_container_width=True
            ):

                def get_selected_llamafile_endpoint() -> str:
                    try:
                        return st.session_state.chat_config_list[0].get("base_url")
                    except:
                        return oai_model_config_selector(
                            st.session_state.oai_like_model_config_dict
                        )[1]

                llamafile_endpoint = st.text_input(
                    label=i18n("Llamafile endpoint"),
                    value=get_selected_llamafile_endpoint(),
                    key="llamafile_endpoint",
                    type="password",
                )

                def get_selected_llamafile_api_key() -> str:
                    try:
                        return st.session_state.chat_config_list[0].get("api_key")
                    except:
                        return oai_model_config_selector(
                            st.session_state.oai_like_model_config_dict
                        )[2]

                llamafile_api_key = st.text_input(
                    label=i18n("Llamafile API key"),
                    value=get_selected_llamafile_api_key(),
                    key="llamafile_api_key",
                    type="password",
                    placeholder=i18n("Fill in your API key. (Optional)"),
                )

                def save_oai_like_config_button_callback():
                    config_id = oailike_config_processor.update_config(
                        model=select_box1,
                        base_url=llamafile_endpoint,
                        api_key=llamafile_api_key,
                        description=st.session_state.get("config_description", "")
                    )
                    st.toast(i18n("Model config saved successfully"), icon="✅")
                    return config_id

                config_description = st.text_input(
                    label=i18n("Config Description"),
                    key="config_description",
                    placeholder=i18n("Enter a description for this configuration")
                )

                save_oai_like_config_button = st.button(
                    label=i18n("Save model config"),
                    on_click=save_oai_like_config_button_callback,
                    use_container_width=True,
                )

                st.write("---")

                config_list = oailike_config_processor.list_model_configs()
                config_options = [f"{config['model']} - {config['description']}" for config in config_list]
                
                selected_config = st.selectbox(
                    label=i18n("Select model config"),
                    options=config_options,
                    format_func=lambda x: x,
                    on_change=lambda: st.toast(
                        i18n("Click the Load button to apply the configuration"),
                        icon="🚨",
                    ),
                    key="selected_config"
                )

                def load_oai_like_config_button_callback():
                    selected_index = config_options.index(st.session_state.selected_config)
                    selected_config_id = config_list[selected_index]['id']
                    
                    logger.info(f"Loading model config: {selected_config_id}")
                    config = oailike_config_processor.get_model_config(config_id=selected_config_id)
                    
                    if config:
                        config_data = config
                        st.session_state.oai_like_model_config_dict = {config_data['model']: config_data}
                        st.session_state.rag_current_run_id_index = rag_run_id_list.index(
                            st.session_state.rag_run_id
                        )
                        st.session_state.model = config_data['model']
                        st.session_state.llamafile_endpoint = config_data['base_url']
                        st.session_state.llamafile_api_key = config_data['api_key']
                        st.session_state.config_description = config_data.get('description', '')
                        
                        logger.info(
                            f"Llamafile Model config loaded: {st.session_state.oai_like_model_config_dict}"
                        )

                        # 更新rag_chat_config_list
                        st.session_state["rag_chat_config_list"][0]["model"] = config_data['model']
                        st.session_state["rag_chat_config_list"][0]["api_key"] = config_data['api_key']
                        st.session_state["rag_chat_config_list"][0]["base_url"] = config_data['base_url']

                        logger.info(
                            f"Chat config list updated: {st.session_state.rag_chat_config_list}"
                        )
                        chat_history_storage.upsert(
                            AssistantRun(
                                run_id=st.session_state.rag_run_id,
                                llm=st.session_state["rag_chat_config_list"][0],
                                assistant_data={
                                    "model_type": st.session_state["model_type"],
                                    # "system_prompt": st.session_state["system_prompt"],
                                },
                                updated_at=datetime.now(),
                            )
                        )
                        st.toast(i18n("Model config loaded successfully"), icon="✅")
                    else:
                        st.toast(i18n("Failed to load model config"), icon="❌")

                load_oai_like_config_button = st.button(
                    label=i18n("Load model config"),
                    use_container_width=True,
                    type="primary",
                    on_click=load_oai_like_config_button_callback,
                )

                def delete_oai_like_config_button_callback():
                    selected_index = config_options.index(st.session_state.selected_config)
                    selected_config_id = config_list[selected_index]['id']
                    oailike_config_processor.delete_model_config(selected_config_id)
                    st.toast(i18n("Model config deleted successfully"), icon="🗑️")
                    # st.rerun()

                delete_oai_like_config_button = st.button(
                    label=i18n("Delete model config"),
                    use_container_width=True,
                    on_click=delete_oai_like_config_button_callback,
                )

        reset_model_button = model_choosing_container.button(
            label=i18n("Reset model info"),
            on_click=lambda x: x.cache_clear(),
            args=(model_selector,),
            use_container_width=True,
        )

    with rag_knowledge_base_settings_tab:
        with st.container(border=True):

            def update_collection_processor_callback():
                with open(embedding_config_file_path, "r", encoding="utf-8") as f:
                    embedding_config = json.load(f)

                st.session_state.collection_config = next(
                    (
                        kb
                        for kb in embedding_config.get("knowledge_bases", [])
                        if kb["name"] == st.session_state["collection_name"]
                    ),
                    {},
                )

                st.session_state.reset_counter += 1

            def get_collection_options():
                try:
                    with open(embedding_config_file_path, "r", encoding="utf-8") as f:
                        embedding_config = json.load(f)
                    return [
                        kb["name"] for kb in embedding_config.get("knowledge_bases", [])
                    ]
                except FileNotFoundError:
                    logger.error(f"File not found: {embedding_config_file_path}")
                    return []

            collection_selectbox = st.selectbox(
                label=i18n("Collection"),
                options=get_collection_options(),
                placeholder=i18n("Please create a new collection first"),
                on_change=update_collection_processor_callback,
                key="collection_name",
            )

            collection_files_placeholder = st.empty()
            refresh_collection_files_button_placeholder = st.empty()

            query_mode_toggle = st.toggle(
                label=i18n("Single file query mode"),
                value=False,
                help=i18n(
                    "Default is whole collection query mode, if enabled, the source document would only be the selected file"
                ),
                on_change=update_collection_processor_callback,
            )

            if collection_selectbox and query_mode_toggle:
                with open(embedding_config_file_path, "r", encoding="utf-8") as f:
                    embedding_config = json.load(f)
                st.session_state.collection_config = next(
                    (
                        kb
                        for kb in embedding_config.get("knowledge_bases", [])
                        if kb["name"] == st.session_state["collection_name"]
                    ),
                    {},
                )
                collection_processor = ChromaCollectionProcessorWithNoApi(
                    collection_name=st.session_state["collection_name"],
                    embedding_config=EmbeddingConfiguration(**embedding_config),
                    embedding_model_id=st.session_state.collection_config.get(
                        "embedding_model_id"
                    ),
                )

                selected_collection_file = collection_files_placeholder.selectbox(
                    label=i18n("Files"),
                    options=collection_processor.list_all_filechunks_raw_metadata_name(
                        st.session_state.reset_counter
                    ),
                    format_func=lambda x: x.split("/")[-1].split("\\")[-1],
                )

                def refresh_collection_files_button_callback():
                    st.session_state.reset_counter += 1
                    refresh_retriever()
                    st.toast("知识库文件已刷新")

                refresh_collection_files_button = (
                    refresh_collection_files_button_placeholder.button(
                        label=i18n("Refresh files"),
                        use_container_width=True,
                        on_click=refresh_collection_files_button_callback,
                    )
                )
            else:
                selected_collection_file = None

            is_rerank = st.toggle(label=i18n("Rerank"), value=False, key="is_rerank")
            is_hybrid_retrieve = st.toggle(
                label=i18n("Hybrid retrieve"), value=False, key="is_hybrid_retrieve"
            )
            hybrid_retrieve_weight_placeholder = st.empty()
            if is_hybrid_retrieve:
                hybrid_retrieve_weight = hybrid_retrieve_weight_placeholder.slider(
                    label=i18n("Hybrid retrieve weight"),
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="hybrid_retrieve_weight",
                )
            else:
                # Prevent error when the checkbox is unchecked
                hybrid_retrieve_weight = 0.0

    delete_previous_round_button_col, clear_button_col = rag_dialog_settings_tab.columns(2)

    def clear_chat_history_callback():
        st.session_state.custom_rag_chat_history = []
        st.session_state.custom_rag_sources = {}
        chat_history_storage.upsert(
            AssistantRun(
                name="assistant",
                run_id=st.session_state.rag_run_id,
                run_name=st.session_state.rag_run_name,
                memory={"chat_history": st.session_state.custom_rag_chat_history},
                task_data={"source_documents": st.session_state.custom_rag_sources},
            )
        )
        st.session_state.rag_current_run_id_index = rag_run_id_list.index(
            st.session_state.rag_run_id
        )
        st.toast(body=i18n("Chat history cleared"), icon="🧹")

    def delete_previous_round_callback():
        st.session_state.custom_rag_chat_history = st.session_state.custom_rag_chat_history[:-2]

        # 删除最后一轮对话对应的源文档
        if st.session_state.custom_rag_chat_history:
            last_message = st.session_state.custom_rag_chat_history[-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                st.session_state.custom_rag_sources = {key: value for key, value in st.session_state.custom_rag_sources.items() if key not in last_message['content']}
            else:
                logger.warning("Final message format error, cannot delete corresponding source documents")
        else:
            logger.info("Chat history is empty, no need to delete source documents")

        chat_history_storage.upsert(
            AssistantRun(
                name="assistant",
                run_id=st.session_state.rag_run_id,
                run_name=st.session_state.rag_run_name,
                memory={"chat_history": st.session_state.custom_rag_chat_history},
            )
        )

    delete_previous_round_button = delete_previous_round_button_col.button(
        label=i18n("Delete previous round"),
        on_click=delete_previous_round_callback,
        use_container_width=True,
    )

    clear_button = clear_button_col.button(
        label=i18n("Clear chat history"),
        on_click=clear_chat_history_callback,
        use_container_width=True,
    )

    export_button = st.button(
        label=i18n("Export chat history"),
        use_container_width=True,
    )
    if export_button:
        export_dialog(
            chat_history=st.session_state.custom_rag_chat_history, 
            is_rag=True,
            chat_name=st.session_state.rag_run_name,
            model_name=st.session_state.model
        )

    back_to_top_placeholder0 = st.empty()
    back_to_top_placeholder1 = st.empty()
    back_to_top_bottom_placeholder0 = st.empty()
    back_to_top_bottom_placeholder1 = st.empty()


float_init()
# st.write(st.session_state.rag_chat_config_list)
st.title(st.session_state.rag_run_name)
write_custom_rag_chat_history(
    st.session_state.custom_rag_chat_history, st.session_state.custom_rag_sources
)
back_to_top(back_to_top_placeholder0, back_to_top_placeholder1)
back_to_bottom(back_to_top_bottom_placeholder0, back_to_top_bottom_placeholder1)
# Control the chat input to prevent error when the model is not selected
if st.session_state.model == None:
    st.session_state.prompt_disabled = True
else:
    st.session_state.prompt_disabled = False
prompt = float_chat_input_with_audio_recorder(
    if_tools_call=False, prompt_disabled=st.session_state.prompt_disabled
)

if prompt and st.session_state.model != None:
    with st.chat_message("user", avatar=user_avatar):
        st.html("<span class='rag-chat-user'></span>")
        st.markdown(prompt)
        st.html(RAG_CHAT_USER_STYLE)

    # Add user message to chat history
    st.session_state.custom_rag_chat_history.append({"role": "user", "content": prompt})

    processed_messages = list_length_transform(
        history_length, st.session_state.custom_rag_chat_history
    )
    # 在 invoke 的 messages 中去除 response_id
    processed_messages = [
        dict_filter(item, ["role", "content"]) for item in processed_messages
    ]

    with st.chat_message("assistant", avatar=ai_avatar):
        with st.spinner("Thinking..."):
            refresh_retriever()
            agentchat_processor = get_agentchat_processor()
            response = agentchat_processor.create_custom_rag_response(
                collection_name=collection_selectbox,
                messages=processed_messages,
                is_rerank=is_rerank,
                is_hybrid_retrieve=is_hybrid_retrieve,
                hybrid_retriever_weight=hybrid_retrieve_weight,
                stream=if_stream,
                selected_file=selected_collection_file,
            )
        st.html("<span class='rag-chat-assistant'></span>")
        handle_response(response=response, if_stream=if_stream)
        st.html(RAG_CHAT_ASSISTANT_STYLE)

elif st.session_state.model == None:
    st.error(i18n("Please select a model"))
