import streamlit as st

from autogen.agentchat.contrib.capabilities import transforms

import os
from typing import Optional
from uuid import uuid4
from copy import deepcopy
from dotenv import load_dotenv
load_dotenv()

from api.dependency import APIRequestHandler,SUPPORTED_SOURCES

from llm.aoai.completion import aoai_config_generator
from llm.ollama.completion import ollama_config_generator
from llm.groq.completion import groq_config_generator
from llm.llamafile.completion import llamafile_config_generator
from configs.basic_config import I18nAuto,set_pages_configs_in_common,SUPPORTED_LANGUAGES
from configs.chat_config import ChatProcessor, OAILikeConfigProcessor
from utils.basic_utils import model_selector, save_basic_chat_history, oai_model_config_selector, write_chat_history
from storage.db.sqlite import SqlAssistantStorage
from model.chat.assistant import AssistantRun


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

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

VERSION = "0.0.1"
logo_path = os.path.join(os.path.dirname(__file__), "img", "RAGenT_logo.png")
set_pages_configs_in_common(
    version=VERSION,
    title="RAGenT",
    page_icon_path=logo_path
)


# Initialize chat history
if "chat_history" not in st.session_state:
    try:
        st.session_state.chat_history = deepcopy(st.session_state.chat_history_list.memory['chat_history'])
    except:
        st.session_state.chat_history = []

# Initialize openai-like model config
if "oai_like_model_config_dict" not in st.session_state:
    st.session_state.oai_like_model_config_dict = {
        "noneed":{
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "noneed"
        }
    }

if "run_id" not in st.session_state:
    try:
        st.session_state.run_id = st.session_state.chat_history_list.memory.run_id
    except:
        st.session_state.run_id = str(uuid4())

try:
    run_id_list = chat_history_storage.get_all_run_ids()
    if st.session_state.run_id in run_id_list:
        st.session_state.current_run_id_index = run_id_list.index(st.session_state.chat_history_list.memory.run_id)
    else:
        st.session_state.current_run_id_index = 0
except:
    st.session_state.current_run_id_index = 0

with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

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
                st.rerun()

            delete_oai_like_config_button = st.button(
                label=i18n("Delete model config"),
                use_container_width=True,
                on_click=oailike_config_processor.delete_model_config,
                args=(oai_like_config_list,)
            )

    with st.expander(label=i18n("Model config")):
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

    dialog_settings = st.popover(
        label=i18n("Saved dialog settings"),
        use_container_width=True,
        # disabled=True,
    )

    history_length = st.number_input(
        label=i18n("History length"),
        min_value=1,
        value=16,
        step=1,
        key="history_length"
    )
    # 根据历史对话消息数，创建 MessageHistoryLimiter 
    max_msg_transfrom = transforms.MessageHistoryLimiter(max_messages=history_length)

    cols = st.columns(2)
    export_button = cols[0].button(label=i18n("Export chat history"),use_container_width=True)
    clear_button = cols[1].button(label=i18n("Clear chat history"),use_container_width=True)
    # 本来这里是放clear_button的，但是因为需要更新current_run_id_index，所以放在了下面
    if export_button:
        # 将聊天历史导出为Markdown
        chat_history = "\n".join([f"# {message['role']} \n\n{message['content']}\n\n" for message in st.session_state.chat_history])
        # st.markdown(chat_history)
        # 将Markdown保存到本地文件夹中
        with open("chat_history.md", "w") as f:
            f.write(chat_history)
        st.toast(body="Chat history exported to chat_history.md",icon="🎉")

    def get_system_prompt(run_id: Optional[str]):
        if run_id:
            try:
                return chat_history_storage.get_specific_run(run_id).assistant_data['system_prompt']
            except:
                return "You are a helpful assistant."
        else:
            return "You are a helpful assistant."

    dialog_settings.write(i18n("Dialogues list"))
    
    # 管理已有对话
    dialogs_container = dialog_settings.container(height=200,border=True)
    saved_dialog = dialogs_container.radio(
        label=i18n("Saved dialog"),
        options=chat_history_storage.get_all_runs(),
        format_func=lambda x: x.run_name,
        index=st.session_state.current_run_id_index,
        label_visibility="collapsed",
        key="chat_history_list",
    )
    add_dialog_button = dialog_settings.button(
        label=i18n("Add a new dialog"),
        use_container_width=True,
    )
    delete_dialog_button = dialog_settings.button(
        label=i18n("Delete selected dialog"),
        use_container_width=True,
    )

    if saved_dialog:
        st.session_state.run_id = saved_dialog.run_id
        st.session_state.chat_history = chat_history_storage.get_specific_run(saved_dialog.run_id).memory["chat_history"]
    if add_dialog_button:
        chat_history_storage.upsert(
            AssistantRun(
                name="assistant",
                run_id=str(uuid4()),
                run_name="New dialog",
                memory={
                    "chat_history": []
                }
            )
        )
        st.rerun()
    if delete_dialog_button:
        chat_history_storage.delete_run(st.session_state.run_id)
        st.session_state.run_id = str(uuid4())
        st.session_state.chat_history = []
        st.rerun()
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

    dialog_settings.write("---")

    # 保存对话
    def get_run_name():
        try:
            run_name = saved_dialog.run_name
        except:
            run_name = "RAGenT"
        return run_name
    def change_run_name():
        # TODO: 修改对话名称
        pass
    dialog_name = dialog_settings.text_input(
        label=i18n("Dialog name"),
        value=get_run_name(),
        key="run_name",
    )
    
    dialog_settings.text_area(
        label=i18n("System Prompt"),
        value=get_system_prompt(saved_dialog.run_id),
        height=100,
    )

if st.session_state["model_type"] == "OpenAI":
    pass
elif st.session_state["model_type"] == "AOAI":
    config_list = aoai_config_generator(
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        stream = if_stream,
    )
elif st.session_state["model_type"] == "Ollama":
    config_list = ollama_config_generator(
        model = st.session_state["model"],
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        stream = if_stream,
    )
elif st.session_state["model_type"] == "Groq":
    config_list = groq_config_generator(
        model = st.session_state["model"]
    )
elif st.session_state["model_type"] == "Llamafile":
    # 避免因为API_KEY为空字符串导致的请求错误（500）
    if st.session_state["llamafile_api_key"] == "":
        custom_api_key = "noneed"
    else:
        custom_api_key = st.session_state["llamafile_api_key"]
    config_list = llamafile_config_generator(
        model = st.session_state["model"],
        base_url = st.session_state["llamafile_endpoint"],
        api_key = custom_api_key,
        max_tokens = max_tokens,
        temperature = temperature,
        top_p = top_p,
        stream = if_stream,
    )


st.title(st.session_state.run_name)
write_chat_history(st.session_state.chat_history)


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
            # 对消息的数量进行限制
            processed_messages = max_msg_transfrom.apply_transform(deepcopy(st.session_state.chat_history))
            processed_messages.insert(0,{"role": "system", "content": st.session_state.system_prompt})

            chatprocessor = ChatProcessor(
                requesthandler=requesthandler,
                model_type=st.session_state["model_type"],
                llm_config=config_list[0],
            )

            # 非流式调用
            if not config_list[0].get("params",{}).get("stream",False):

                # 如果 model_type 的小写名称在 SUPPORTED_SOURCES 字典中才响应
                # 一般都是在的
                response = chatprocessor.create_completion_noapi(
                    messages=processed_messages
                )

                if "error" not in response:
                    # st.write(response)
                    response_content = response.choices[0].message.content
                    st.write(response_content)
                    cost = response.cost
                    st.write(f"response cost: ${cost}")

                    st.session_state.chat_history.append({"role": "assistant", "content": response_content})    

                    # 保存聊天记录
                    chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_name=st.session_state.run_name,
                            run_id=st.session_state.run_id,
                            llm=config_list[0],
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

                response = chatprocessor.create_completion_stream_noapi(
                    messages=processed_messages
                )
                total_response = st.write_stream(response)

                st.session_state.chat_history.append({"role": "assistant", "content": total_response})
                chat_history_storage.upsert(
                        AssistantRun(
                            name="assistant",
                            run_id=st.session_state.run_id,
                            run_name=st.session_state.run_name,
                            llm=config_list[0],
                            memory={
                                "chat_history": st.session_state.chat_history
                            },
                            assistant_data={
                                "system_prompt": st.session_state.system_prompt,
                            }
                        )
                    )
        # TODO：没有添加直接修改run_name的功能前，先使用rerun更新
        st.rerun()