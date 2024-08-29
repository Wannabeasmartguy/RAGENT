import streamlit as st
from streamlit_float import *

from autogen.agentchat.contrib.capabilities import transforms

import os
from datetime import datetime
from typing import Optional
from uuid import uuid4
from copy import deepcopy
from loguru import logger
from utils.log.logger_config import setup_logger
from dotenv import load_dotenv
load_dotenv(override=True)

from api.dependency import APIRequestHandler

from llm.aoai.completion import aoai_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.groq.completion import groq_openai_config_generator
from llm.llamafile.completion import llamafile_config_generator
from configs.basic_config import I18nAuto,set_pages_configs_in_common,SUPPORTED_LANGUAGES
from configs.chat_config import ChatProcessor, OAILikeConfigProcessor
from utils.basic_utils import model_selector, oai_model_config_selector, write_chat_history, config_list_postprocess
try:
    from utils.st_utils import float_chat_input_with_audio_recorder, back_to_top, back_to_bottom
except:
    st.rerun()
from storage.db.sqlite import SqlAssistantStorage
from model.chat.assistant import AssistantRun
from utils.chat.prompts import ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT
from tools.toolkits import filter_out_selected_tools_dict, filter_out_selected_tools_list


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])

requesthandler = APIRequestHandler("localhost", os.getenv("SERVER_PORT",8000))

oailike_config_processor = OAILikeConfigProcessor()

chat_history_db_dir = os.path.join(os.path.dirname(__file__), "databases", "chat_history")
chat_history_db_file = os.path.join(chat_history_db_dir, "chat_history.db")
if not os.path.exists(chat_history_db_dir):
    os.makedirs(chat_history_db_dir)
chat_history_storage = SqlAssistantStorage(
    table_name="chatbot_chat_history",
    db_file = chat_history_db_file,
)
if not chat_history_storage.table_exists():
    chat_history_storage.create()


VERSION = "0.1.1"
logo_path = os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png")
# Solve set_pages error caused by "Go to top/bottom of page" button.
# Only need st.rerun once to fix it, and it works fine thereafter.
try:
    set_pages_configs_in_common(
        version=VERSION,
        title="RAGenT",
        page_icon_path=logo_path
    )
except:
    st.rerun()

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed":{
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "noneed"
        }
    }

run_id_list = chat_history_storage.get_all_run_ids()
if len(run_id_list) == 0:
    chat_history_storage.upsert(
        AssistantRun(
            name="assistant",
            run_id=str(uuid4()),
            llm=aoai_config_generator()[0],
            run_name="New dialog",
            memory={
                "chat_history": []
            },
            assistant_data={
                "model_type": "AOAI"
            }
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
    st.session_state.chat_config_list = [chat_history_storage.get_specific_run(st.session_state.run_id).llm]
# initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = chat_history_storage.get_specific_run(st.session_state.run_id).memory["chat_history"]

def update_config_in_db_callback():
    if st.session_state["model_type"] == "OpenAI":
        pass
    elif st.session_state["model_type"] == "AOAI":
        config_list = aoai_config_generator(
            model = st.session_state.model,
            max_tokens = st.session_state.max_tokens,
            temperature = st.session_state.temperature,
            top_p = st.session_state.top_p,
            stream = st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Ollama":
        config_list = ollama_config_generator(
            model = st.session_state.model,
            max_tokens = st.session_state.max_tokens,
            temperature = st.session_state.temperature,
            top_p = st.session_state.top_p,
            stream = st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Groq":
        config_list = groq_openai_config_generator(
            model = st.session_state.model,
            max_tokens = st.session_state.max_tokens,
            temperature = st.session_state.temperature,
            top_p = st.session_state.top_p,
            stream = st.session_state.if_stream,
        )
    elif st.session_state["model_type"] == "Llamafile":
        try:
            config_list = llamafile_config_generator(
                model = st.session_state.model,
                api_key = st.session_state.llamafile_api_key,
                base_url = st.session_state.llamafile_endpoint,
                max_tokens = st.session_state.max_tokens,
                temperature = st.session_state.temperature,
                top_p = st.session_state.top_p,
                stream = st.session_state.if_stream,
            )
        except (UnboundLocalError,AttributeError):
            config_list = llamafile_config_generator()
    st.session_state["chat_config_list"] = config_list
    chat_history_storage.upsert(
        AssistantRun(
            run_id=st.session_state.run_id,
            llm=config_list[0],
            assistant_data={
                "model_type": st.session_state["model_type"],
                "system_prompt": st.session_state["system_prompt"],
            },
            updated_at=datetime.now()
        )
    )

with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    # st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

    dialog_settings_tab, model_settings_tab= st.tabs([i18n("Dialog Settings"), i18n("Model Settings")])

    with model_settings_tab:
        model_choosing_container = st.expander(label=i18n("Model Choosing"),expanded=True)
        def get_model_type_index():
            options = ["AOAI","OpenAI","Ollama","Groq","Llamafile"]
            try:
                return options.index(chat_history_storage.get_specific_run(st.session_state.run_id).assistant_data["model_type"])
            except:
                return 0
        select_box0 = model_choosing_container.selectbox(
            label=i18n("Model type"),
            options=["AOAI","OpenAI","Ollama","Groq","Llamafile"],
            index=get_model_type_index(),
            key="model_type",
            on_change=update_config_in_db_callback
        )

        with st.expander(label=i18n("Model config"),expanded=True):
            max_tokens = st.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get("max_tokens", 1900),
                step=1,
                key="max_tokens",
                on_change=update_config_in_db_callback,
                help=i18n("Maximum number of tokens to generate in the completion.Different models may have different constraints, e.g., the Qwen series of models require a range of [0,2000).")
            )
            temperature = st.slider(
                label=i18n("Temperature"),
                min_value=0.0,
                max_value=2.0,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get("temperature", 0.5),
                step=0.1,
                key="temperature",
                on_change=update_config_in_db_callback,
                help=i18n("'temperature' controls the randomness of the model. Lower values make the model more deterministic and conservative, while higher values make it more creative and diverse. The default value is 0.5.")
            )
            top_p = st.slider(
                label=i18n("Top p"),
                min_value=0.0,
                max_value=1.0,
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get("top_p", 0.5),
                step=0.1,
                key="top_p",
                on_change=update_config_in_db_callback,
                help=i18n("Similar to 'temperature', but don't change it at the same time as temperature")
            )
            if_stream = st.toggle(
                label=i18n("Stream"),
                value=config_list_postprocess(st.session_state.chat_config_list)[0].get("stream", True),
                key="if_stream",
                on_change=update_config_in_db_callback,
                help=i18n("Whether to stream the response as it is generated, or to wait until the entire response is generated before returning it. Default is False, which means to wait until the entire response is generated before returning it.")
            )
            if_tools_call = st.toggle(
                label=i18n("Tools call"),
                value=False,
                key="if_tools_call",
                help=i18n("Whether to enable the use of tools. Only available for some models. For unsupported models, normal chat mode will be used by default."),
                on_change=lambda: logger.info(f"Tools call toggled, current status: {str(st.session_state.if_tools_call)}")
            )
        
        # 为了让 update_config_in_db_callback 能够更新上面的多个参数，需要把model选择放在他们下面
        if select_box0 != "Llamafile":
            def get_selected_non_llamafile_model_index(model_type) -> int:
                try:
                    return model_selector(model_type).index(st.session_state.chat_config_list[0].get("model"))
                except:
                    return None
            select_box1 = model_choosing_container.selectbox(
                label=i18n("Model"),
                options=model_selector(st.session_state["model_type"]),
                index=get_selected_non_llamafile_model_index(st.session_state["model_type"]),
                key="model",
                on_change=update_config_in_db_callback
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
                    options=oailike_config_processor.get_config(),
                    on_change=lambda: st.toast(i18n("Click the Load button to apply the configuration"),icon="🚨"),
                )
                def load_oai_like_config_button_callback():
                    st.session_state.oai_like_model_config_dict = oailike_config_processor.get_model_config(oai_like_config_list)
                    st.session_state.current_run_id_index = run_id_list.index(st.session_state.run_id)
                load_oai_like_config_button = st.button(
                    label=i18n("Load model config"),
                    use_container_width=True,
                    type="primary",
                    on_click=load_oai_like_config_button_callback
                )
                if load_oai_like_config_button:
                    model_config = next(iter(st.session_state.oai_like_model_config_dict.values()))
                    config_list = llamafile_config_generator(
                        model=next(iter(st.session_state.oai_like_model_config_dict.keys())),
                        api_key=model_config.get("api_key"),
                        base_url=model_config.get("base_url")
                    )
                    st.session_state["chat_config_list"] = config_list
                    chat_history_storage.upsert(
                        AssistantRun(
                            run_id=st.session_state.run_id,
                            llm=config_list[0],
                            assistant_data={
                                "model_type": st.session_state["model_type"],
                                "system_prompt": st.session_state["system_prompt"],
                            },
                            updated_at=datetime.now()
                        )
                    )
                    # toast here doesn't work
                    st.toast(i18n("Model config loaded successfully"))

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

    with dialog_settings_tab:
        def get_system_prompt(run_id: Optional[str]):
            if run_id:
                try:
                    return chat_history_storage.get_specific_run(run_id).assistant_data['system_prompt']
                except:
                    return "You are a helpful assistant."
            else:
                return "You are a helpful assistant."
        
        st.write(i18n("Dialogues list"))
        
        # 管理已有对话
        dialogs_container = st.container(height=250,border=True)
        def saved_dialog_change_callback():
            st.session_state.run_id = st.session_state.saved_dialog.run_id
            st.session_state.chat_config_list = [chat_history_storage.get_specific_run(st.session_state.saved_dialog.run_id).llm]
            try:
                st.session_state.chat_history = chat_history_storage.get_specific_run(st.session_state.saved_dialog.run_id).memory["chat_history"]
            except:
                st.session_state.chat_history = []
        saved_dialog = dialogs_container.radio(
            label=i18n("Saved dialog"),
            options=chat_history_storage.get_all_runs(),
            format_func=lambda x: x.run_name[:15]+'...' if len(x.run_name) > 15 else x.run_name,
            index=st.session_state.current_run_id_index,
            label_visibility="collapsed",
            key="saved_dialog",
            on_change=saved_dialog_change_callback
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
                        llm=st.session_state.chat_config_list[0],
                        memory={
                            "chat_history": []
                        },
                        assistant_data={
                            "system_prompt": get_system_prompt(st.session_state.run_id),
                        }
                    )
                )
            add_dialog_button = st.button(
                label=i18n("Add a new dialog"),
                use_container_width=True,
                on_click=add_dialog_button_callback
            )
        with delete_dialog_column:
            def delete_dialog_callback():
                chat_history_storage.delete_run(st.session_state.run_id)
                if len(chat_history_storage.get_all_run_ids()) == 0:
                    chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_id=st.session_state.run_id,
                            run_name="New dialog",
                            memory={
                                "chat_history": []
                            },
                            assistant_data={
                                "system_prompt": get_system_prompt(st.session_state.run_id),
                            }
                        )
                    )
                    st.session_state.chat_history = []
                else:
                    st.session_state.run_id = chat_history_storage.get_all_run_ids()[0]
                # st.rerun()
            delete_dialog_button = st.button(
                label=i18n("Delete selected dialog"),
                use_container_width=True,
                on_click=delete_dialog_callback
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
        def dialog_name_change_callback():
            chat_history_storage.upsert(
                AssistantRun(
                    run_name=st.session_state.run_name,
                    run_id=st.session_state.run_id,
                )
            )
            st.session_state.current_run_id_index = run_id_list.index(st.session_state.run_id)
        dialog_name = dialog_details_settings_popover.text_input(
            label=i18n("Dialog name"),
            value=get_run_name(),
            key="run_name",
            on_change=dialog_name_change_callback
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
                        "system_prompt": st.session_state.system_prompt
                    }
                )    
            )
        )
        history_length = dialog_details_settings_popover.number_input(
            label=i18n("History length"),
            min_value=1,
            value=16,
            step=1,
            key="history_length"
        )

    # 根据历史对话消息数，创建 MessageHistoryLimiter 
    max_msg_transfrom = transforms.MessageHistoryLimiter(max_messages=history_length)

    export_button_col, clear_button_col = st.columns(2)
    export_button = export_button_col.button(label=i18n("Export chat history"),use_container_width=True)
    clear_button = clear_button_col.button(label=i18n("Clear chat history"),use_container_width=True)
    # 本来这里是放clear_button的，但是因为需要更新current_run_id_index，所以放在了下面
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
        # st.markdown(chat_history)
        # 将Markdown保存到本地文件夹中
        with open("chat_history.md", "w") as f:
            f.write(chat_history)
        st.toast(body="Chat history exported to chat_history.md",icon="🎉")
    if clear_button:
        st.session_state.chat_history = []
        chat_history_storage.upsert(
            AssistantRun(
                name="assistant",
                run_id=st.session_state.run_id,
                run_name=st.session_state.run_name,
                memory={
                    "chat_history": st.session_state.chat_history
                }
            )
        )
        st.session_state.current_run_id_index = run_id_list.index(st.session_state.run_id)
        st.rerun()
    
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
prompt = float_chat_input_with_audio_recorder(if_tools_call=if_tools_call)
# # st.write(filter_out_selected_tools_list(st.session_state.tools_popover))
# st.write(filter_out_selected_tools_dict(st.session_state.tools_popover))

# Accept user input
if prompt and st.session_state.model != None:
# if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 对消息的数量进行限制
            processed_messages = max_msg_transfrom.apply_transform(deepcopy(st.session_state.chat_history))

            # 如果是工具调用，则将用户输入的 system prompt 并入工具调用系统提示，否则直接使用用户输入的 system prompt
            processed_messages.insert(0, {"role": "system", "content": ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT.format(user_system_prompt=st.session_state.system_prompt) if if_tools_call else st.session_state.system_prompt})

            chatprocessor = ChatProcessor(
                requesthandler=requesthandler,
                model_type=st.session_state["model_type"],
                llm_config=st.session_state.chat_config_list[0],
            )

            # 非流式调用
            if not st.session_state.chat_config_list[0].get("params",{}).get("stream",False):

                # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中才响应
                # 一般都是在的
                if not if_tools_call:
                    response = chatprocessor.create_completion_noapi(
                        messages=processed_messages
                    )
                else:
                    tools_list_selected = filter_out_selected_tools_list(st.session_state.tools_popover)
                    tools_map_selected = filter_out_selected_tools_dict(st.session_state.tools_popover)
                    logger.debug(f"tools_list_selected: {tools_list_selected}")
                    logger.debug(f"tools_map_selected: {tools_map_selected}")
                    response = chatprocessor.create_tools_call_completion(
                        messages=processed_messages,
                        tools=tools_list_selected,
                        function_map=tools_map_selected
                    )

                if "error" not in response:
                    # st.write(response)
                    response_content = response.choices[0].message.content
                    st.write(response_content)
                    try:
                        cost = response.cost
                        st.write(f"response cost: ${cost}")
                    except:
                        pass

                    st.session_state.chat_history.append({"role": "assistant", "content": response_content})    

                    # 保存聊天记录
                    chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_name=st.session_state.run_name,
                            run_id=st.session_state.run_id,
                            llm=st.session_state.chat_config_list[0],
                            memory={
                                "chat_history": st.session_state.chat_history
                            },
                        )
                    )
                else:
                    st.error(response)
 
            else:
                # 流式调用
                # 获得 API 的响应，但是解码出来的乱且不完整
                # response = chatprocessor.create_completion_stream_api(
                #     messages=processed_messages
                # )
                # for chunk in response:
                #     st.write(chunk.decode("utf-8","ignore"))
                #     time.sleep(0.1)

                if not if_tools_call:
                    response = chatprocessor.create_completion_stream_noapi(
                        messages=processed_messages
                    )
                else:
                    tools_list_selected = filter_out_selected_tools_list(st.session_state.tools_popover)
                    tools_map_selected = filter_out_selected_tools_dict(st.session_state.tools_popover)
                    logger.debug(f"tools_list_selected: {tools_list_selected}")
                    logger.debug(f"tools_map_selected: {tools_map_selected}")
                    response = chatprocessor.create_tools_call_completion(
                        messages=processed_messages,
                        tools=tools_list_selected,
                        function_map=tools_map_selected
                    )
                total_response = st.write_stream(response)

                st.session_state.chat_history.append({"role": "assistant", "content": total_response})
                chat_history_storage.upsert(
                    AssistantRun(
                        name="assistant",
                        run_id=st.session_state.run_id,
                        run_name=st.session_state.run_name,
                        llm=st.session_state.chat_config_list[0],
                        memory={
                            "chat_history": st.session_state.chat_history
                        },
                        assistant_data={
                            "system_prompt": st.session_state.system_prompt,
                        }
                    )
                )
elif st.session_state.model == None:
    st.error("Please select a model")