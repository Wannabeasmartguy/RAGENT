import streamlit as st
import os
from uuid import uuid4
from loguru import logger
from datetime import datetime
from pydantic import ValidationError

from configs.basic_config import (
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
)

from configs.chat_config import AgentChatProcessor, OAILikeConfigProcessor
from configs.knowledge_base_config import ChromaVectorStoreProcessor
from api.dependency import APIRequestHandler
from storage.db.sqlite import SqlAssistantStorage
from model.chat.assistant import AssistantRun


@st.cache_data
def write_custom_rag_chat_history(chat_history, _sources):
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # st.markdown("**Source**:")
                # 展示引用源
                row1 = st.columns(3)
                row2 = st.columns(3)

                response_sources_list = _sources[message["response_id"]]

                for index, pop in enumerate(row1 + row2):
                    a = pop.popover(i18n("Cited Source") + f" {index+1}", use_container_width=True)
                    file_name = response_sources_list["metadatas"][index]["source"]
                    file_content = response_sources_list["page_content"][index]
                    a.text(i18n("Cited Source")+ ": " + file_name)
                    a.code(file_content, language="plaintext")


requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT", 8000))

oailike_config_processor = OAILikeConfigProcessor()

vectorstore_processor = ChromaVectorStoreProcessor(
    # 仅需要展示所有的 Collection 即可，故所有参数都为空
    embedding_model_name_or_path="",
    embedding_model_type="huggingface",
)

VERSION = "0.1.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, "img", "RAGenT_logo.png")
logo_text = os.path.join(parent_directory, "img", "RAGenT_logo_with_text_horizon.png")

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
            llm=aoai_config_generator(model=None)[0],
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


def update_rag_config_in_db_callback():
    """
    Update rag chat llm config in db.
    """
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
        except (UnboundLocalError, AttributeError):
            config_list = llamafile_config_generator()
    elif st.session_state["model_type"] == "LiteLLM":
        config_list = litellm_config_generator(model=st.session_state["model"])
    st.session_state["rag_chat_config_list"] = config_list
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


set_pages_configs_in_common(version=VERSION, title="RAG Chat", page_icon_path=logo_path)

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
    st.write("---")

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
            st.session_state.rag_run_id = st.session_state.rag_saved_dialog.run_id
            st.session_state.rag_chat_config_list = [
                chat_history_storage.get_specific_run(
                    st.session_state.rag_saved_dialog.run_id
                ).llm
            ]
            try:
                st.session_state.rag_chat_history = (
                    chat_history_storage.get_specific_run(
                        st.session_state.rag_saved_dialog.run_id
                    ).memory["chat_history"]
                )
            except:
                st.session_state.rag_chat_history = []

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
                        llm=aoai_config_generator(model=None)[0],
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
                            llm=aoai_config_generator(model=None)[0],
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

            delete_dialog_button = st.button(
                label=i18n("Delete selected dialog"),
                use_container_width=True,
                on_click=delete_rag_dialog_callback,
            )

        if saved_dialog:
            st.session_state.rag_run_id = saved_dialog.run_id
            try:
                st.session_state.custom_rag_chat_history = (
                    chat_history_storage.get_specific_run(saved_dialog.run_id).memory[
                        "chat_history"
                    ]
                )
                st.session_state.custom_rag_sources = (
                    chat_history_storage.get_specific_run(
                        saved_dialog.run_id
                    ).task_data["source_documents"]
                )
            except (TypeError, ValidationError):
                st.session_state.custom_rag_chat_history = []
                st.session_state.custom_rag_sources = {}

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
            chat_history_storage.upsert(
                AssistantRun(
                    run_name=st.session_state.rag_run_name,
                    run_id=st.session_state.rag_run_id,
                )
            )
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

        history_length = st.number_input(
            label=i18n("History length"),
            min_value=1,
            value=32,
            step=1,
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
                max_value=2.0,
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
            if_tools_call = st.toggle(
                label=i18n("Tools call"),
                value=False,
                key="if_tools_call",
                help=i18n(
                    "Whether to enable the use of tools. Only available for some models. For unsupported models, normal chat mode will be used by default."
                ),
                on_change=lambda: logger.info(
                    f"Tools call toggled, current status: {str(st.session_state.if_tools_call)}"
                ),
            )

        # 为了让 update_config_in_db_callback 能够更新上面的多个参数，需要把model选择放在他们下面
        if select_box0 != "Llamafile":
            def get_selected_non_llamafile_model_index(model_type) -> int:
                try:
                    return model_selector(model_type).index(
                        st.session_state.rag_chat_config_list[0].get("model")
                    )
                except:
                    return None

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
                try:
                    return st.session_state.chat_config_list[0].get("model")
                except:
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
                    placeholder=i18n("Fill in your API key. (Optional)"),
                )
                save_oai_like_config_button = st.button(
                    label=i18n("Save model config"),
                    on_click=oailike_config_processor.update_config,
                    args=(select_box1, llamafile_endpoint, llamafile_api_key),
                    use_container_width=True,
                )

                st.write("---")

                oai_like_config_list = st.selectbox(
                    label=i18n("Select model config"),
                    options=oailike_config_processor.get_config(),
                    on_change=lambda: st.toast(
                        i18n("Click the Load button to apply the configuration"),
                        icon="🚨",
                    ),
                )

                def load_oai_like_config_button_callback():
                    st.session_state.oai_like_model_config_dict = (
                        oailike_config_processor.get_model_config(oai_like_config_list)
                    )
                    st.session_state.rag_current_run_id_index = rag_run_id_list.index(
                        st.session_state.rag_run_id
                    )
                    st.session_state.model = next(
                        iter(st.session_state.oai_like_model_config_dict.keys())
                    )
                    st.session_state.llamafile_endpoint = next(
                        iter(st.session_state.oai_like_model_config_dict.values())
                    ).get("base_url")
                    st.session_state.llamafile_api_key = next(
                        iter(st.session_state.oai_like_model_config_dict.values())
                    ).get("api_key")

                load_oai_like_config_button = st.button(
                    label=i18n("Load model config"),
                    use_container_width=True,
                    type="primary",
                    on_click=load_oai_like_config_button_callback,
                )
                if load_oai_like_config_button:
                    model_config = next(
                        iter(st.session_state.oai_like_model_config_dict.values())
                    )
                    config_list = llamafile_config_generator(
                        model=next(
                            iter(st.session_state.oai_like_model_config_dict.keys())
                        ),
                        api_key=model_config.get("api_key"),
                        base_url=model_config.get("base_url"),
                    )
                    st.session_state["rag_chat_config_list"] = config_list
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
                    # toast here doesn't work
                    st.toast(i18n("Model config loaded successfully"))

                delete_oai_like_config_button = st.button(
                    label=i18n("Delete model config"),
                    use_container_width=True,
                    on_click=oailike_config_processor.delete_model_config,
                    args=(oai_like_config_list,),
                )

        reset_model_button = model_choosing_container.button(
            label=i18n("Reset model info"),
            on_click=lambda x: x.cache_clear(),
            args=(model_selector,),
            use_container_width=True,
        )

    with rag_knowledge_base_settings_tab:
        with st.popover(label=i18n("RAG Setting"), use_container_width=True):
            collection_selectbox = st.selectbox(
                label=i18n("Collection"),
                options=vectorstore_processor.list_all_knowledgebase_collections(1),
            )
            is_rerank = st.checkbox(label=i18n("Rerank"), value=False, key="is_rerank")
            is_hybrid_retrieve = st.checkbox(
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

    export_button_col, clear_button_col = st.columns(2)
    export_button = export_button_col.button(
        label=i18n("Export chat history"), use_container_width=True
    )
    clear_button = clear_button_col.button(
        label=i18n("Clear chat history"), use_container_width=True
    )
    if clear_button:
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
        st.rerun()
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join(
            [
                f"# {message['role']} \n\n{message['content']}\n\n"
                for message in st.session_state.custom_rag_chat_history
            ]
        )
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
        st.toast(body=i18n(f"Chat history exported to {filename}"), icon="🎉")


agentchat_processor = AgentChatProcessor(
    requesthandler=requesthandler,
    model_type=select_box0,
    llm_config=st.session_state.rag_chat_config_list[0],
)

# st.write(st.session_state.rag_chat_config_list)
st.title(st.session_state.rag_run_name)
write_custom_rag_chat_history(
    st.session_state.custom_rag_chat_history, st.session_state.custom_rag_sources
)

# Control the chat input to prevent error when the model is not selected
if st.session_state.model == None:
    st.session_state.prompt_disabled = True
else:
    st.session_state.prompt_disabled = False
prompt = st.chat_input("What is up?", disabled=st.session_state.prompt_disabled)

if prompt and st.session_state.model != None:
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.custom_rag_chat_history.append({"role": "user", "content": prompt})

    processed_messages = list_length_transform(
        history_length, st.session_state.custom_rag_chat_history
    )
    # 在 invoke 的 messages 中去除 response_id
    processed_messages = [
        dict_filter(item, ["role", "content"]) for item in processed_messages
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agentchat_processor.create_custom_rag_response(
                collection_name=collection_selectbox,
                messages=processed_messages,
                is_rerank=is_rerank,
                is_hybrid_retrieve=is_hybrid_retrieve,
                hybrid_retriever_weight=hybrid_retrieve_weight,
                stream=if_stream,
            )

        # 流式输出
        if if_stream:
            # 先将引用sources添加到 st.session
            st.session_state.custom_rag_sources.update(
                {response.response_id: response.source_documents}
            )

            # 展示回答
            answer = st.write_stream(response.answer)

            # 添加回答到 st.session
            st.session_state.custom_rag_chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "response_id": response.response_id,
                }
            )

            # 保存聊天记录
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
                    }
                )
            )

            # 展示引用源
            row1 = st.columns(3)
            row2 = st.columns(3)

            response_sources = st.session_state.custom_rag_sources[response.response_id]

            for index, pop in enumerate(row1 + row2):
                a = pop.popover(f"引用文件", use_container_width=True)
                file_name = response_sources["metadatas"][index]["source"]
                file_content = response_sources["page_content"][index]
                a.text(f"引用文件{file_name}")
                a.code(file_content, language="plaintext")

        # 非流式输出
        else:
            response = response.model_dump()
            # 将回答添加入 st.sesstion
            st.session_state.custom_rag_chat_history.append(
                {
                    "role": "assistant",
                    "content": response["answer"]["choices"][0]["message"]["content"],
                    "response_id": response["response_id"],
                }
            )

            # 将引用sources添加到 st.session
            st.session_state.custom_rag_sources.update(
                {response["response_id"]: response["source_documents"]}
            )

            # 展示回答
            st.write(response["answer"]["choices"][0]["message"]["content"])

            # 保存聊天记录
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
                    }
                )
            )

            # 展示引用源
            row1 = st.columns(3)
            row2 = st.columns(3)

            response_sources_list = st.session_state.custom_rag_sources[
                response["response_id"]
            ]

            for index, pop in enumerate(row1 + row2):
                a = pop.popover(f"引用文件", use_container_width=True)
                file_name = response_sources_list["metadatas"][index]["source"]
                file_content = response_sources_list["page_content"][index]
                a.text(f"引用文件{file_name}")
                a.code(file_content, language="plaintext")
elif st.session_state.model == None:
    st.error(i18n("Please select a model"))
