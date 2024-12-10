import streamlit as st
import os
import asyncio
from typing import List, Union, Coroutine, AsyncGenerator
from copy import deepcopy
from loguru import logger

from core.basic_config import (
    I18nAuto,
    set_pages_configs_in_common,
)
from utils.basic_utils import (
    model_selector,
)
from config.constants import (
    VERSION,
    LOGO_DIR,
)
from core.llm._client_info import generate_client_config
from core.processors.config.llm import OAILikeConfigProcessor
from config.constants.i18n import I18N_DIR, SUPPORTED_LANGUAGES
from utils.basic_utils import config_list_postprocess, oai_model_config_selector
from ext.autogen.teams.reflect import ReflectionTeamBuilder

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, MultiModalMessage


def update_config_in_db_callback():
    """
    Update config in db.
    """
    origin_config_list = deepcopy(st.session_state.agent_chat_config_list)

    st.session_state.model = model_selector(st.session_state["model_type"])[0]
    if st.session_state["model_type"] == "Llamafile":
        # 先获取模型配置
        model_config = oailike_config_processor.get_model_config(
            model=st.session_state.model
        )
        if model_config and len(model_config) > 1:
            for model_id, model_config_detail in model_config.items():
                if st.session_state.model in model_config_detail.get("model"):
                    selected_model_config = model_config_detail
                    break
        elif model_config and len(model_config) == 1:
            selected_model_config = next(iter(model_config.values()))
        else:
            selected_model_config = {}

        config_list = [
            generate_client_config(
                source=st.session_state["model_type"].lower(),
                model=(
                    st.session_state.model
                    if selected_model_config and len(selected_model_config) > 0  # 检查配置是否存在且非空
                    else "Not given"
                ),
                api_key=selected_model_config.get("api_key", "Not given"),
                base_url=selected_model_config.get("base_url", "Not given"),
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                max_tokens=st.session_state.max_tokens,
                stream=st.session_state.if_stream,
            ).model_dump()
        ]
    else:
        config_list = [
            generate_client_config(
                source=st.session_state["model_type"].lower(),
                model=st.session_state.model,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                max_tokens=st.session_state.max_tokens,
                stream=st.session_state.if_stream,
            ).model_dump()
        ]
    st.session_state["agent_chat_config_list"] = config_list


def create_and_run_reflection_team(user_task: TextMessage):
    reflection_team = (
        ReflectionTeamBuilder()
        .set_model_client(
            source=st.session_state["model_type"].lower(), 
            config_list=st.session_state.agent_chat_config_list
        ).set_primary_agent(
            system_message="You are a helpful AI assistant."
        ).set_critic_agent(
            system_message="Provide constructive feedback. Respond with 'APPROVE' to when your feedbacks are addressed."
        ).set_max_messages(
            max_messages=5
        ).set_termination_text(
            text="APPROVE"
        ).build()
    )
    if st.session_state.if_stream:
        response = reflection_team.run_stream(task=user_task)
        return response
    else:
        response = reflection_team.run(task=user_task)
        return response


def write_task_result(
        task_result: TaskResult, 
        *,
        with_final_answer: bool = True,
        with_thought: bool = True
):
    with st.chat_message(name="assistant thought", avatar="🤖"):
        if with_thought:
            with st.expander(label="Thought", expanded=True):
                for message in task_result.messages:
                    with st.container(border=True):
                        if isinstance(message, TextMessage):
                            st.write(f"{message.source}: ")
                            st.write(message.content)
    if with_final_answer:
        with st.chat_message(name="assistant", avatar="🤖"):
            content = [message.content for message in task_result.messages if message.content != "" and message.content != "APPROVE"]
            st.write(content[-1])


async def write_coroutine(
        agent_chat_result: Coroutine, 
        *,
        with_final_answer: bool = True,
        with_thought: bool = True
    ):
    # 如果是协程，先执行协程以获取结果
    result = await agent_chat_result
    if isinstance(result, TaskResult):
        write_task_result(result, with_final_answer=with_final_answer, with_thought=with_thought)
        return result


async def write_stream_result(
    agent_chat_result: AsyncGenerator, 
    *,
    with_final_answer: bool = True,
    with_thought: bool = True
):
    generator = agent_chat_result
    last_result = None
    
    if with_thought:
        with st.chat_message(name="assistant", avatar="🤖"):
            with st.expander(label="Thought", expanded=True):
                async for chunk in generator:
                    if isinstance(chunk, TextMessage):
                        with st.container(border=True):
                            st.write(f"{chunk.source}: ")
                            st.write(chunk.content)
                    elif isinstance(chunk, TaskResult):
                        last_result = chunk

    # 在循环结束后处理最后的 TaskResult
    if last_result and with_final_answer:
        with st.chat_message(name="assistant", avatar="🤖"):
            content = [message.content for message in last_result.messages if message.content != "" and message.content != "APPROVE"]
            st.write(content[-1])
    
    return last_result


async def write_chunks_or_coroutine(
        agent_chat_result: Union[Coroutine, AsyncGenerator],
        *,
        with_final_answer: bool = True,
        with_thought: bool = True
    ):
    # 检查 agent_chat_result 是协程还是异步生成器
    # 如果是协程，可知使用了run
    if asyncio.iscoroutine(agent_chat_result):
        return await write_coroutine(agent_chat_result, with_final_answer=with_final_answer, with_thought=with_thought)
    elif isinstance(agent_chat_result, AsyncGenerator):
        # 如果是异步生成器，可知使用了run_stream
        return await write_stream_result(agent_chat_result, with_final_answer=with_final_answer, with_thought=with_thought)
    else:
        raise ValueError("Invalid agent chat result type")


def write_chat_history(chat_history: List[Union[TextMessage,TaskResult]]):
    for message in chat_history:
        if isinstance(message, TaskResult):
            write_task_result(message)
        elif isinstance(message, TextMessage):
            if message.source == "user":
                with st.chat_message(name="user", avatar="🧑‍💻"):
                    st.write(message.content)

oailike_config_processor = OAILikeConfigProcessor()

language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES[language]
)

# initialize config
if "agent_chat_config_list" not in st.session_state:
    st.session_state.agent_chat_config_list = [generate_client_config(
        source="openai",
        model=model_selector("OpenAI")[0]
    ).model_dump()]
# initialize chat history
if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []


logo_path = os.path.join(LOGO_DIR, "RAGenT_logo.png")
logo_text = os.path.join(LOGO_DIR, "RAGenT_logo_with_text_horizon.png")
set_pages_configs_in_common(
    version=VERSION, title="RAGenT-AgentChat", page_icon_path=logo_path
)


with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")
    st.write(i18n("Sub pages"))
    st.page_link(
        "pages/2_📖Knowledge_Base_Setting.py", label=(i18n("📖 Knowledge Base Setting"))
    )
    st.write("---")

    dialog_settings_tab, model_settings_tab, multimodal_settings_tab = st.tabs(
        [i18n("Dialog Settings"), i18n("Model Settings"), i18n("Multimodal Settings")],
    )

    with model_settings_tab:
        model_choosing_container = st.expander(
            label=i18n("Model Choosing"), expanded=True
        )

        select_box0 = model_choosing_container.selectbox(
            label=i18n("Model type"),
            options=["OpenAI", "Ollama", "Groq", "Llamafile"],
            key="model_type",
            on_change=update_config_in_db_callback,
        )

        with st.expander(label=i18n("Model config"), expanded=True):
            max_tokens = st.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=config_list_postprocess(st.session_state.agent_chat_config_list)[0].get(
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
                value=config_list_postprocess(st.session_state.agent_chat_config_list)[0].get(
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
                value=config_list_postprocess(st.session_state.agent_chat_config_list)[0].get(
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
                value=config_list_postprocess(st.session_state.agent_chat_config_list)[0].get(
                    "stream", True
                ),
                key="if_stream",
                on_change=update_config_in_db_callback,
                help=i18n(
                    "Whether to stream the response as it is generated, or to wait until the entire response is generated before returning it. Default is False, which means to wait until the entire response is generated before returning it."
                ),
            )

        # 为了让 update_config_in_db_callback 能够更新上面的多个参数，需要把model选择放在他们下面
        if select_box0 != "Llamafile":

            def get_selected_non_llamafile_model_index(model_type) -> int:
                try:
                    model = st.session_state.agent_chat_config_list[0].get("model")
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
                            st.session_state.agent_chat_config_list[0].update(
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
                key="model",
                on_change=update_config_in_db_callback,
            )
        elif select_box0 == "Llamafile":

            def get_selected_llamafile_model() -> str:
                if st.session_state.agent_chat_config_list:
                    return st.session_state.agent_chat_config_list[0].get("model")
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
                        return st.session_state.agent_chat_config_list[0].get("base_url")
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
                        return st.session_state.agent_chat_config_list[0].get("api_key")
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
                        description=st.session_state.get("config_description", ""),
                    )
                    st.toast(i18n("Model config saved successfully"), icon="✅")
                    return config_id

                config_description = st.text_input(
                    label=i18n("Config Description"),
                    key="config_description",
                    placeholder=i18n("Enter a description for this configuration"),
                )

                save_oai_like_config_button = st.button(
                    label=i18n("Save model config"),
                    on_click=save_oai_like_config_button_callback,
                    use_container_width=True,
                )

                st.write("---")

                config_list = oailike_config_processor.list_model_configs()
                config_options = [
                    f"{config['model']} - {config['description']}"
                    for config in config_list
                ]

                selected_config = st.selectbox(
                    label=i18n("Select model config"),
                    options=config_options,
                    format_func=lambda x: x,
                    on_change=lambda: st.toast(
                        i18n("Click the Load button to apply the configuration"),
                        icon="🚨",
                    ),
                    key="selected_config",
                )

                def load_oai_like_config_button_callback():
                    selected_index = config_options.index(
                        st.session_state.selected_config
                    )
                    selected_config_id = config_list[selected_index]["id"]

                    logger.info(f"Loading model config: {selected_config_id}")
                    config = oailike_config_processor.get_model_config(
                        config_id=selected_config_id
                    )

                    if config:
                        config_data = config  # 不再需要 next(iter(config.values()))
                        st.session_state.oai_like_model_config_dict = {
                            config_data["model"]: config_data
                        }
                        st.session_state.model = config_data["model"]
                        st.session_state.llamafile_endpoint = config_data["base_url"]
                        st.session_state.llamafile_api_key = config_data["api_key"]
                        st.session_state.config_description = config_data.get(
                            "description", ""
                        )

                        logger.info(
                            f"Llamafile Model config loaded: {st.session_state.oai_like_model_config_dict}"
                        )

                        # 更新chat_config_list
                        st.session_state["agent_chat_config_list"][0]["model"] = config_data["model"]
                        st.session_state["agent_chat_config_list"][0]["api_key"] = config_data["api_key"]
                        st.session_state["agent_chat_config_list"][0]["base_url"] = config_data["base_url"]

                        logger.info(
                            f"Chat config list updated: {st.session_state.agent_chat_config_list}"
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
                    selected_index = config_options.index(
                        st.session_state.selected_config
                    )
                    selected_config_id = config_list[selected_index]["id"]
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
        st.write(st.session_state.agent_chat_config_list)

write_chat_history(st.session_state.agent_chat_history)

if prompt := st.chat_input(placeholder="Enter your message here"):
    # 用户输入
    user_task = TextMessage(source="user", content=prompt)
    st.session_state.agent_chat_history.append(user_task)
    with st.chat_message(name="user", avatar="🧑‍💻"):
        st.write(user_task.content)
    
    # 思考
    with st.spinner(text="Thinking..."):
        response = create_and_run_reflection_team(user_task)
    
        # 输出
        try:
            result = asyncio.run(write_chunks_or_coroutine(response))
            if result and isinstance(result, TaskResult):
                st.session_state.agent_chat_history.append(result)
        except Exception as e:
            st.error(f"Error writing response: {e}")
            result = response
