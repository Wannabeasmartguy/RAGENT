import streamlit as st
from streamlit_float import *

from autogen.agentchat.contrib.capabilities import transforms

import os
import base64
from datetime import datetime
from typing import Optional, List, Dict, Union
from uuid import uuid4
from copy import deepcopy
from loguru import logger
from utils.log.logger_config import setup_logger, log_dict_changes
from dotenv import load_dotenv

load_dotenv(override=True)

from api.dependency import APIRequestHandler

from llm.oai.completion import oai_config_generator
from llm.aoai.completion import aoai_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.groq.completion import groq_openai_config_generator
from llm.llamafile.completion import llamafile_config_generator
from core.basic_config import (
    I18nAuto,
    set_pages_configs_in_common,
    SUPPORTED_LANGUAGES,
)
from core.chat_processors import ChatProcessor, OAILikeConfigProcessor
from utils.basic_utils import (
    model_selector,
    oai_model_config_selector,
    write_chat_history,
    config_list_postprocess,
    user_input_constructor,
    get_style,
    USER_AVATAR_SVG,
    AI_AVATAR_SVG
)

try:
    from utils.st_utils import (
        float_chat_input_with_audio_recorder,
        back_to_top,
        back_to_bottom,
        export_dialog,
    )
except:
    st.rerun()
from storage.db.sqlite import SqlAssistantStorage
from model.chat.assistant import AssistantRun
from utils.chat.prompts import ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT
from tools.toolkits import (
    filter_out_selected_tools_dict,
    filter_out_selected_tools_list,
)


def generate_response(
    *,
    processed_messages: List[Dict[str, Union[str, Dict, List]]],
    chatprocessor: ChatProcessor,
    if_tools_call: bool,
    if_stream: bool,
):
    if if_tools_call:
        tools_list_selected = filter_out_selected_tools_list(
            st.session_state.tools_popover
        )
        tools_map_selected = filter_out_selected_tools_dict(
            st.session_state.tools_popover
        )
        logger.debug(f"tools_list_selected: {tools_list_selected}")
        logger.debug(f"tools_map_selected: {tools_map_selected}")
        response = chatprocessor.create_tools_call_completion(
            messages=processed_messages,
            tools=tools_list_selected,
            function_map=tools_map_selected,
        )
    else:
        if if_stream:
            response = chatprocessor.create_completion_stream_noapi(
                messages=processed_messages
            )
        else:
            response = chatprocessor.create_completion_noapi(
                messages=processed_messages
            )
    return response


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])

requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT", 8000))

oailike_config_processor = OAILikeConfigProcessor()

chat_history_db_dir = os.path.join(
    os.path.dirname(__file__), "databases", "chat_history"
)
chat_history_db_file = os.path.join(chat_history_db_dir, "chat_history.db")
if not os.path.exists(chat_history_db_dir):
    os.makedirs(chat_history_db_dir)
chat_history_storage = SqlAssistantStorage(
    table_name="chatbot_chat_history",
    db_file=chat_history_db_file,
)
if not chat_history_storage.table_exists():
    chat_history_storage.create()


VERSION = "0.1.1"
logo_path = os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png")
logo_text = os.path.join(
    os.path.dirname(__file__), "img", "RAGenT_logo_with_text_horizon.png"
)
# 将SVG编码为base64
user_avatar = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"
ai_avatar = f"data:image/svg+xml;base64,{base64.b64encode(AI_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"

# Solve set_pages error caused by "Go to top/bottom of page" button.
# Only need st.rerun once to fix it, and it works fine thereafter.
try:
    set_pages_configs_in_common(
        version=VERSION, title="RAGenT", page_icon_path=logo_path
    )
except:
    st.rerun()


# ********** Initialize session state **********

# 回调函数防抖
if 'last_dialog_change_time' not in st.session_state:
    st.session_state.last_dialog_change_time = 0
if 'debounce_delay' not in st.session_state:
    st.session_state.debounce_delay = 0.5  # 500毫秒的防抖延迟

if "prompt_disabled" not in st.session_state:
    st.session_state.prompt_disabled = False

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed": {"base_url": "http://127.0.0.1:8080/v1", "api_key": "noneed"}
    }

run_id_list = chat_history_storage.get_all_run_ids()
if len(run_id_list) == 0:
    chat_history_storage.upsert(
        AssistantRun(
            name="assistant",
            run_id=str(uuid4()),
            llm=aoai_config_generator()[0],
            run_name="New dialog",
            memory={"chat_history": []},
            assistant_data={"model_type": "AOAI"},
        )
    )
    run_id_list = chat_history_storage.get_all_run_ids()

if "current_run_id_index" not in st.session_state:
    st.session_state.current_run_id_index = 0
while st.session_state.current_run_id_index > len(run_id_list):
    st.session_state.current_run_id_index -= 1
if "run_id" not in st.session_state:
    st.session_state.run_id = run_id_list[st.session_state.current_run_id_index]

# initialize config
if "chat_config_list" not in st.session_state:
    st.session_state.chat_config_list = [
        chat_history_storage.get_specific_run(st.session_state.run_id).llm
    ]
# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = chat_history_storage.get_specific_run(
        st.session_state.run_id
    ).memory["chat_history"]


def debounced_dialog_change():
    """
    防抖函数，用于防止对话框频繁切换时，将其他对话的配置更新到当前对话中。
    经测试必须要在当前文件定义，否则无法正常工作。
    """
    import time
    current_time = time.time()
    if current_time - st.session_state.last_dialog_change_time > st.session_state.debounce_delay:
        st.session_state.last_dialog_change_time = current_time
        return True
    return False

def update_config_in_db_callback():
    """
    Update config in db.
    """
    origin_config_list = deepcopy(st.session_state.chat_config_list)
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
    st.session_state["chat_config_list"] = config_list
    log_dict_changes(original_dict=origin_config_list[0], new_dict=config_list[0])
    chat_history_storage.upsert(
        AssistantRun(
            run_id=st.session_state.run_id,
            llm=config_list[0],
            assistant_data={
                "model_type": st.session_state["model_type"],
                "system_prompt": st.session_state["system_prompt"],
            },
            updated_at=datetime.now(),
        )
    )


with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    # st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

    dialog_settings_tab, model_settings_tab, multimodal_settings_tab = st.tabs(
        [i18n("Dialog Settings"), i18n("Model Settings"), i18n("Multimodal Settings")],
    )

    with model_settings_tab:
        model_choosing_container = st.expander(
            label=i18n("Model Choosing"), expanded=True
        )

        def get_model_type_index():
            options = ["AOAI", "OpenAI", "Ollama", "Groq", "Llamafile"]
            try:
                return options.index(
                    chat_history_storage.get_specific_run(
                        st.session_state.run_id
                    ).assistant_data["model_type"]
                )
            except:
                return 0

        select_box0 = model_choosing_container.selectbox(
            label=i18n("Model type"),
            options=["AOAI", "OpenAI", "Ollama", "Groq", "Llamafile"],
            index=get_model_type_index(),
            key="model_type",
            on_change=update_config_in_db_callback,
        )

        with st.expander(label=i18n("Model config"), expanded=True):
            max_tokens = st.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get(
                    "max_tokens", 1900
                ),
                step=1,
                key="max_tokens",
                on_change=update_config_in_db_callback,
                help=i18n(
                    "Maximum number of tokens to generate in the completion.Different models may have different constraints, e.g., the Qwen series of models require a range of [0,2000)."
                ),
            )
            temperature = st.slider(
                label=i18n("Temperature"),
                min_value=0.0,
                max_value=1.0,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get(
                    "temperature", 0.5
                ),
                step=0.1,
                key="temperature",
                on_change=update_config_in_db_callback,
                help=i18n(
                    "'temperature' controls the randomness of the model. Lower values make the model more deterministic and conservative, while higher values make it more creative and diverse. The default value is 0.5."
                ),
            )
            top_p = st.slider(
                label=i18n("Top p"),
                min_value=0.0,
                max_value=1.0,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get(
                    "top_p", 0.5
                ),
                step=0.1,
                key="top_p",
                on_change=update_config_in_db_callback,
                help=i18n(
                    "Similar to 'temperature', but don't change it at the same time as temperature"
                ),
            )
            if_stream = st.toggle(
                label=i18n("Stream"),
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get(
                    "stream", True
                ),
                key="if_stream",
                on_change=update_config_in_db_callback,
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
                    model = st.session_state.chat_config_list[0].get("model")
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
                            st.session_state.chat_config_list[0].update(
                                {"model": options[0]}
                            )
                            logger.debug(
                                f"model {model} not in options, set model in config list to first option: {options[0]}"
                            )
                            return 0
                except (ValueError, AttributeError, IndexError):
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
                on_change=update_config_in_db_callback,
            )
        elif select_box0 == "Llamafile":

            def get_selected_llamafile_model() -> str:
                if st.session_state.chat_config_list:
                    return st.session_state.chat_config_list[0].get("model")
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
                        config_data = config  # 不再需要 next(iter(config.values()))
                        st.session_state.oai_like_model_config_dict = {config_data['model']: config_data}
                        st.session_state.current_run_id_index = run_id_list.index(
                            st.session_state.run_id
                        )
                        st.session_state.model = config_data['model']
                        st.session_state.llamafile_endpoint = config_data['base_url']
                        st.session_state.llamafile_api_key = config_data['api_key']
                        st.session_state.config_description = config_data.get('description', '')
                        
                        logger.info(
                            f"Llamafile Model config loaded: {st.session_state.oai_like_model_config_dict}"
                        )

                        # 更新chat_config_list
                        st.session_state["chat_config_list"][0]["model"] = config_data['model']
                        st.session_state["chat_config_list"][0]["api_key"] = config_data['api_key']
                        st.session_state["chat_config_list"][0]["base_url"] = config_data['base_url']

                        logger.info(
                            f"Chat config list updated: {st.session_state.chat_config_list}"
                        )
                        chat_history_storage.upsert(
                            AssistantRun(
                                run_id=st.session_state.run_id,
                                llm=st.session_state["chat_config_list"][0],
                                assistant_data={
                                    "model_type": st.session_state["model_type"],
                                    "system_prompt": st.session_state["system_prompt"],
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

    with dialog_settings_tab:

        def get_system_prompt(run_id: Optional[str]):
            if run_id:
                try:
                    return chat_history_storage.get_specific_run(run_id).assistant_data[
                        "system_prompt"
                    ]
                except:
                    return "You are a helpful assistant."
            else:
                return "You are a helpful assistant."

        st.write(i18n("Dialogues list"))

        # 管理已有对话
        dialogs_container = st.container(height=250, border=True)

        def saved_dialog_change_callback():
            if debounced_dialog_change():
                # 获取当前选中的对话
                selected_run = st.session_state.saved_dialog

                # 更新session state
                st.session_state.run_id = selected_run.run_id
                st.session_state.current_run_id_index = chat_history_storage.get_all_run_ids().index(selected_run.run_id)
                
                # 获取并更新chat_config_list
                new_chat_config = selected_run.llm
                st.session_state.chat_config_list = [new_chat_config] if new_chat_config else []

                logger.info(f"Chat dialog changed, selected dialog name: {selected_run.run_name}, selected dialog id: {selected_run.run_id}")
                
                # 如果需要记录变化，可以保留这部分
                if 'chat_config_list' in st.session_state and st.session_state.chat_config_list and new_chat_config:
                    log_dict_changes(
                        original_dict=st.session_state.chat_config_list[0],
                        new_dict=new_chat_config,
                    )

                # 更新聊天历史
                st.session_state.chat_history = selected_run.memory.get("chat_history", []) if selected_run.memory else []

                # 更新其他相关的session state
                st.session_state.model_type = selected_run.assistant_data.get("model_type") if selected_run.assistant_data else None
                st.session_state.system_prompt = selected_run.assistant_data.get("system_prompt") if selected_run.assistant_data else None

            else:
                st.toast(i18n("Please wait, processing the last dialog switch..."), icon="🔄")

        saved_dialog = dialogs_container.radio(
            label=i18n("Saved dialog"),
            options=chat_history_storage.get_all_runs(),
            format_func=lambda x: (
                x.run_name[:15] + "..." if len(x.run_name) > 15 else x.run_name
            ),
            index=st.session_state.current_run_id_index,
            label_visibility="collapsed",
            key="saved_dialog",
            on_change=saved_dialog_change_callback,
        )

        add_dialog_column, delete_dialog_column = st.columns([1, 1])
        with add_dialog_column:

            def add_dialog_button_callback():
                st.session_state.run_id = str(uuid4())
                chat_history_storage.upsert(
                    AssistantRun(
                        name="assistant",
                        run_id=st.session_state.run_id,
                        run_name="New dialog",
                        llm=aoai_config_generator(
                            model=model_selector("AOAI")[0], stream=True
                        )[0],
                        memory={"chat_history": []},
                        assistant_data={
                            "system_prompt": get_system_prompt(st.session_state.run_id),
                            "model_type": "AOAI",
                        },
                    )
                )
                st.session_state.chat_history = []
                st.session_state.current_run_id_index = 0
                st.session_state.chat_config_list = [
                    chat_history_storage.get_specific_run(st.session_state.run_id).llm
                ]
                logger.info(f"Add a new chat dialog, added dialog name: {st.session_state.run_name}, added dialog id: {st.session_state.run_id}")

            add_dialog_button = st.button(
                label=i18n("Add a new dialog"),
                use_container_width=True,
                on_click=add_dialog_button_callback,
            )
        with delete_dialog_column:

            def delete_dialog_callback():
                chat_history_storage.delete_run(st.session_state.run_id)
                if len(chat_history_storage.get_all_run_ids()) == 0:
                    st.session_state.run_id = str(uuid4())
                    chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_id=st.session_state.run_id,
                            run_name="New dialog",
                            llm=aoai_config_generator(
                                model=model_selector("AOAI")[0], stream=True
                            )[0],
                            memory={"chat_history": []},
                            assistant_data={
                                "system_prompt": get_system_prompt(
                                    st.session_state.run_id
                                ),
                            },
                        )
                    )
                    st.session_state.chat_config_list = [
                        chat_history_storage.get_specific_run(
                            st.session_state.run_id
                        ).llm
                    ]
                    st.session_state.chat_history = []
                else:
                    st.session_state.run_id = chat_history_storage.get_all_run_ids()[0]
                    st.session_state.chat_history = (
                        chat_history_storage.get_specific_run(
                            st.session_state.run_id
                        ).memory["chat_history"]
                    )
                    st.session_state.chat_config_list = [
                        chat_history_storage.get_specific_run(
                            st.session_state.run_id
                        ).llm
                    ]
                logger.info(f"Delete a chat dialog, deleted dialog name: {st.session_state.saved_dialog.run_name}, deleted dialog id: {st.session_state.run_id}")

            delete_dialog_button = st.button(
                label=i18n("Delete selected dialog"),
                use_container_width=True,
                on_click=delete_dialog_callback,
            )

        st.write(i18n("Dialogues details"))

        dialog_details_settings_popover = st.expander(
            label=i18n("Dialogues details"),
            # use_container_width=True
        )

        def dialog_name_change_callback():
            origin_run_name = saved_dialog.run_name
            chat_history_storage.upsert(
                AssistantRun(
                    run_name=st.session_state.run_name,
                    run_id=st.session_state.run_id,
                )
            )
            st.session_state.current_run_id_index = run_id_list.index(
                st.session_state.run_id
            )
            logger.info(f"Chat dialog name changed from {origin_run_name} to {st.session_state.run_name}.(run_id: {st.session_state.run_id})")

        dialog_name = dialog_details_settings_popover.text_input(
            label=i18n("Dialog name"),
            value=chat_history_storage.get_specific_run(st.session_state.run_id).run_name,
            key="run_name",
            on_change=dialog_name_change_callback,
        )

        dialog_details_settings_popover.text_area(
            label=i18n("System Prompt"),
            value=get_system_prompt(st.session_state.run_id),
            height=300,
            key="system_prompt",
            on_change=lambda: chat_history_storage.upsert(
                AssistantRun(
                    run_id=st.session_state.run_id,
                    assistant_data={
                        "model_type": st.session_state.model_type,
                        "system_prompt": st.session_state.system_prompt,
                    },
                )
            ),
        )
        history_length = dialog_details_settings_popover.number_input(
            label=i18n("History length"),
            min_value=1,
            value=16,
            step=1,
            help=i18n(
                "The number of messages to keep in the llm memory."
            ),
            key="history_length",
        )

        # 根据历史对话消息数，创建 MessageHistoryLimiter
        max_msg_transfrom = transforms.MessageHistoryLimiter(
            max_messages=history_length
        )

        delete_previous_round_button_col, clear_button_col = dialog_settings_tab.columns(2)

        def clear_chat_history_callback():
            st.session_state.chat_history = []
            chat_history_storage.upsert(
                AssistantRun(
                    name="assistant",
                    run_id=st.session_state.run_id,
                    run_name=st.session_state.run_name,
                    memory={"chat_history": st.session_state.chat_history},
                )
            )
            st.session_state.current_run_id_index = run_id_list.index(
                st.session_state.run_id
            )
            st.toast(body=i18n("Chat history cleared"), icon="🧹")
        
        def delete_previous_round_callback():
            # 删除最后一轮对话，包含用户消息和助手消息
            st.session_state.chat_history = st.session_state.chat_history[:-2]
            chat_history_storage.upsert(
                AssistantRun(
                    name="assistant",
                    run_id=st.session_state.run_id,
                    run_name=st.session_state.run_name,
                    memory={"chat_history": st.session_state.chat_history},
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
                chat_history=st.session_state.chat_history,
                chat_name=st.session_state.run_name,
                model_name=st.session_state.model
            )

    with multimodal_settings_tab:
        image_uploader = st.file_uploader(
            label=i18n("Upload images"),
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="image_uploader",
        )

    # Fix the bug: "Go to top/bottom of page" cause problem that will make `write_chat_history` can't correctly show the chat history during `write_stream`
    back_to_top_placeholder0 = st.empty()
    back_to_top_placeholder1 = st.empty()
    back_to_top_bottom_placeholder0 = st.empty()
    back_to_top_bottom_placeholder1 = st.empty()


float_init()
st.title(st.session_state.run_name)
write_chat_history(st.session_state.chat_history)
back_to_top(back_to_top_placeholder0, back_to_top_placeholder1)
back_to_bottom(back_to_top_bottom_placeholder0, back_to_top_bottom_placeholder1)
if st.session_state.model == None:
    st.session_state.prompt_disabled = True
else:
    st.session_state.prompt_disabled = False
prompt = float_chat_input_with_audio_recorder(
    if_tools_call=if_tools_call, prompt_disabled=st.session_state.prompt_disabled
)
# # st.write(filter_out_selected_tools_list(st.session_state.tools_popover))
# st.write(filter_out_selected_tools_dict(st.session_state.tools_popover))

# Accept user input
if prompt and st.session_state.model:
    # 显示用户消息
    with st.chat_message("user", avatar=user_avatar):
        st.html("<span class='chat-user'></span>")
        st.markdown(prompt)
        if image_uploader:
            st.image(image_uploader)
        # 根据Streamlit版本选择样式
        st.html(get_style(style_type="USER_CHAT", st_version=st.__version__))

    # Add user message to chat history
    user_input = user_input_constructor(
        prompt=prompt,
        images=image_uploader,
    )
    st.session_state.chat_history.append(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=ai_avatar):
        with st.spinner("Thinking..."):
            # 对消息的数量进行限制
            processed_messages = max_msg_transfrom.apply_transform(
                deepcopy(st.session_state.chat_history)
            )

            system_prompt = (
                ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT.format(
                    user_system_prompt=st.session_state.system_prompt
                )
                if if_tools_call
                else st.session_state.system_prompt
            )
            processed_messages.insert(0, {"role": "system", "content": system_prompt})

            chatprocessor = ChatProcessor(
                requesthandler=requesthandler,
                model_type=st.session_state["model_type"],
                llm_config=st.session_state.chat_config_list[0],
            )

            response = generate_response(
                processed_messages=processed_messages,
                chatprocessor=chatprocessor,
                if_tools_call=if_tools_call,
                if_stream=if_stream,
            )

            st.html("<span class='chat-assistant'></span>")

            if not if_stream:
                if "error" not in response:
                    response_content = response.choices[0].message.content
                    st.write(response_content)
                    st.html(get_style(style_type="ASSISTANT_CHAT", st_version=st.__version__))

                    try:
                        st.write(f"response cost: ${response.cost}")
                    except:
                        pass

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                else:
                    st.error(response)
            else:
                total_response = st.write_stream(response)
                st.html(get_style(style_type="ASSISTANT_CHAT", st_version=st.__version__))
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": total_response}
                )

            # 保存聊天记录
            chat_history_storage.upsert(
                AssistantRun(
                    name="assistant",
                    run_name=st.session_state.run_name,
                    run_id=st.session_state.run_id,
                    llm=st.session_state.chat_config_list[0],
                    memory={"chat_history": st.session_state.chat_history},
                    assistant_data={
                        "system_prompt": st.session_state.system_prompt,
                        "model_type": st.session_state.model_type,
                    },
                )
            )
elif st.session_state.model == None:
    st.error(i18n("Please select a model"))
