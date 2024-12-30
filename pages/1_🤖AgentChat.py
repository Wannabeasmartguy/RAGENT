import streamlit as st
import os
import asyncio
from uuid import uuid4
from typing import List, Union, Coroutine, AsyncGenerator, Literal
from dataclasses import dataclass, asdict
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
    DEFAULT_DIALOG_TITLE,
)
from core.llm._client_info import generate_client_config
from core.processors.config.llm import OAILikeConfigProcessor
from core.processors.dialog.dialog_processors import AgenChatDialogProcessor
from config.constants.i18n import I18N_DIR, SUPPORTED_LANGUAGES
from core.storage.db.sqlite import SqlAssistantStorage
from config.constants import CHAT_HISTORY_DIR, AGENT_CHAT_HISTORY_DB_TABLE, CHAT_HISTORY_DB_FILE
from assets.styles.css.components_css import CUSTOM_RADIO_STYLE
from ext.autogen.teams.reflect import ReflectionTeamBuilder
from ext.autogen.manager.template import AgentTemplateType, AgentTemplateFileManager

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, MultiModalMessage


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


def create_default_dialog(
        dialog_processor: AgenChatDialogProcessor,
        priority: Literal["high", "normal"] = "high",
    ) -> str:
    """
    创建默认对话

    Args:
        dialog_processor: 对话处理器

    Returns:
        对话ID
    """
    from core.processors.dialog.dialog_processors import OperationPriority
    if priority == "high":
        priority = OperationPriority.HIGH
    elif priority == "normal":
        priority = OperationPriority.NORMAL

    new_run_id = str(uuid4())
    try:
        default_template = list(team_template_manager.agent_templates.values())[0]
    except IndexError:
        raise ValueError("No agent templates found")
    dialog_processor.create_dialog(
        run_id=new_run_id,
        run_name=DEFAULT_DIALOG_TITLE,
        llm_config=default_template.get("llm", {}),
        assistant_data={
            "template": default_template,
        },
        priority=priority,
    )
    return new_run_id


@dataclass
class AgentChatState:
    """前端保存的当前AgentChat对话数据"""
    run_id: str
    run_name: str
    chat_history: List[Union[TextMessage,TaskResult]]
    template: dict


if not os.path.exists(CHAT_HISTORY_DIR):
    os.makedirs(CHAT_HISTORY_DIR)
chat_history_storage = SqlAssistantStorage(
    table_name=AGENT_CHAT_HISTORY_DB_TABLE,
    db_file=CHAT_HISTORY_DB_FILE,
)
if not chat_history_storage.table_exists():
    chat_history_storage.create()
dialog_processor = AgenChatDialogProcessor(storage=chat_history_storage)
oailike_config_processor = OAILikeConfigProcessor()
team_template_manager = AgentTemplateFileManager()

language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES[language]
)

run_id_list = [run.run_id for run in dialog_processor.get_all_dialogs()]
if len(run_id_list) == 0:
    create_default_dialog(dialog_processor, priority="normal")
    run_id_list = [run.run_id for run in dialog_processor.get_all_dialogs()]

if "agent_chat_current_run_id_index" not in st.session_state:
    st.session_state.agent_chat_current_run_id_index = 0
while st.session_state.agent_chat_current_run_id_index > len(run_id_list):
    st.session_state.agent_chat_current_run_id_index -= 1
if "agent_chat_run_id" not in st.session_state:
    st.session_state.agent_chat_run_id = run_id_list[st.session_state.agent_chat_current_run_id_index]

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = dialog_processor.get_dialog(
        st.session_state.agent_chat_run_id
    ).memory["chat_history"]

if "agent_chat_current_template" not in st.session_state:
    _template = dialog_processor.get_dialog(
        st.session_state.agent_chat_run_id
    ).assistant_data["template"]
    st.session_state.agent_chat_current_template = _template


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
    # st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")
    st.write(i18n("Sub pages"))
    st.page_link(
        "pages/Agent_Setting.py", label=(i18n("⚙️ Agent Setting"))
    )
    st.write("---")

    dialog_settings_tab, team_settings_tab, multimodal_settings_tab = st.tabs(
        [i18n("Dialog Settings"), i18n("Team Settings"), i18n("Multimodal Settings")],
    )

    with dialog_settings_tab:
        dialogs_list_tab, dialog_details_tab = st.tabs(
            [i18n("Dialogues list"), i18n("Dialogues details")]
        )

        # 管理已有对话
        with dialogs_list_tab:
            dialogs_container = st.container(height=400, border=True)

            def saved_dialog_change_callback():
                try:
                    selected_run = st.session_state.agent_chat_saved_dialog
                    current_run_id = st.session_state.agent_chat_run_id

                    if selected_run.run_id == current_run_id:
                        logger.debug(f"Same dialog selected, skipping update")
                        return

                    # 保存当前对话状态
                    if current_run_id:
                        current_run_state = AgentChatState(
                            run_id=current_run_id,
                            run_name=selected_run.run_name,
                            chat_history=st.session_state.agent_chat_history,
                            template=st.session_state.agent_chat_current_template
                        )
                        dialog_processor.update_dialog_config(
                            run_id=current_run_id,
                            llm_config=current_run_state.template.get("llm", {}),
                            assistant_data={"template": current_run_state.template}
                        )
                    
                    # 加载新对话状态    
                    st.session_state.agent_chat_run_id = selected_run.run_id
                    st.session_state.agent_chat_current_run_id_index = [
                        run.run_id for run in dialog_processor.get_all_dialogs()
                    ].index(st.session_state.agent_chat_run_id)
                    st.session_state.agent_chat_current_template = selected_run.assistant_data["template"]
                    st.session_state.agent_chat_team_template_index = get_team_template_index()
                    st.session_state.agent_chat_history = selected_run.memory["chat_history"]

                    logger.info(f"Chat dialog changed to: {selected_run.run_name} ({st.session_state.agent_chat_run_id})")

                except Exception as e:
                    logger.error(f"Error during dialog change: {e}")
                    st.error(i18n("Failed to change dialog"))

            saved_dialog = dialogs_container.radio(
                label=i18n("Saved dialog"),
                options=dialog_processor.get_all_dialogs(),
                format_func=lambda x: (
                    x.run_name[:15] + "..." if len(x.run_name) > 15 else x.run_name
                ),
                index=st.session_state.agent_chat_current_run_id_index,
                label_visibility="collapsed",
                key="agent_chat_saved_dialog",
                on_change=saved_dialog_change_callback,
            )
            # 自定义radio外观为对话列表卡片样式
            st.markdown(CUSTOM_RADIO_STYLE, unsafe_allow_html=True)

            add_dialog_column, delete_dialog_column = st.columns([1, 1])
            with add_dialog_column:

                def add_dialog_button_callback():
                    new_run_id = create_default_dialog(dialog_processor, priority="normal")
                    new_run = dialog_processor.get_dialog(new_run_id)
                    new_run_state = AgentChatState(
                        run_id=new_run_id,
                        run_name=new_run.run_name,
                        chat_history=new_run.memory.get("chat_history", []),
                        template=new_run.assistant_data["template"],
                    )
                    st.session_state.agent_chat_run_id = new_run_state.run_id
                    st.session_state.agent_chat_run_name = new_run_state.run_name
                    st.session_state.agent_chat_chat_history = new_run_state.chat_history
                    st.session_state.agent_chat_current_template = new_run_state.template
                    st.session_state.agent_chat_current_run_id_index = 0

                    logger.info(
                        f"Add a new chat dialog, added dialog name: {st.session_state.agent_chat_run_name}, added dialog id: {st.session_state.agent_chat_run_id}"
                    )

                add_dialog_button = st.button(
                    label=i18n("Add a new dialog"),
                    use_container_width=True,
                    on_click=add_dialog_button_callback,
                )
            with delete_dialog_column:

                def delete_dialog_callback():
                    dialog_processor.delete_dialog(st.session_state.agent_chat_run_id)
                    if len(dialog_processor.get_all_dialogs()) == 0:
                        st.session_state.agent_chat_run_id = create_default_dialog(dialog_processor, priority="high")
                    else:
                        while st.session_state.agent_chat_current_run_id_index >= len(dialog_processor.get_all_dialogs()):
                            st.session_state.agent_chat_current_run_id_index -= 1
                        st.session_state.agent_chat_run_id = dialog_processor.get_all_dialogs()[
                            st.session_state.agent_chat_current_run_id_index
                        ].run_id
                    current_run = dialog_processor.get_dialog(st.session_state.agent_chat_run_id)
                    st.session_state.agent_chat_history = current_run.memory["chat_history"]
                    logger.info(
                        f"Delete a chat dialog, deleted dialog name: {st.session_state.agent_chat_saved_dialog.run_name}, deleted dialog id: {st.session_state.agent_chat_run_id}"
                    )

                delete_dialog_button = st.button(
                    label=i18n("Delete selected dialog"),
                    use_container_width=True,
                    on_click=delete_dialog_callback,
                )

        with dialog_details_tab:
            dialog_details_settings_popover = st.expander(
                label=i18n("Dialogues details"), expanded=True
            )

            def dialog_name_change_callback():
                """对话名称更改回调"""
                dialog_processor.update_dialog_name(
                    run_id=st.session_state.agent_chat_run_id, new_name=st.session_state.agent_chat_run_name
                )

            dialog_name = dialog_details_settings_popover.text_input(
                label=i18n("Dialog name"),
                value=dialog_processor.get_dialog(st.session_state.agent_chat_run_id).run_name,
                key="agent_chat_run_name",
                on_change=dialog_name_change_callback,
            )

            delete_previous_round_button_col, clear_button_col = (
                dialog_details_tab.columns(2)
            )

            def clear_chat_history_callback():
                st.session_state.agent_chat_chat_history = []
                dialog_processor.update_chat_history(
                    run_id=st.session_state.agent_chat_run_id,
                    chat_history=st.session_state.agent_chat_history,
                )
                st.session_state.agent_chat_current_run_id_index = run_id_list.index(
                    st.session_state.agent_chat_run_id
                )
                st.toast(body=i18n("Chat history cleared"), icon="🧹")

            def delete_previous_round_callback():
                # 删除最后一轮对话
                # 如果前一条是用户消息，后一条是助手消息，则两条都删除
                # 如果后一条是用户消息，则只删除用户消息
                if (
                    len(st.session_state.agent_chat_history) >= 2
                    and st.session_state.agent_chat_history[-1]["role"] == "assistant"
                    and st.session_state.agent_chat_history[-2]["role"] == "user"
                ):
                    st.session_state.agent_chat_history = st.session_state.agent_chat_history[:-2]
                elif len(st.session_state.agent_chat_history) > 0:  # 确保至少有一条消息
                    st.session_state.agent_chat_history = st.session_state.agent_chat_history[:-1]
                dialog_processor.update_chat_history(
                    run_id=st.session_state.agent_chat_run_id,
                    chat_history=st.session_state.agent_chat_history,
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


    with team_settings_tab:

        def team_template_change_callback():
            selected_template = st.session_state.agent_chat_team_template
            st.session_state.agent_chat_current_template = selected_template
            dialog_processor.update_template(
                run_id=st.session_state.agent_chat_run_id,
                template={"template": selected_template}
            )


        def get_team_template_index():
            team_template_dict = team_template_manager.agent_templates
            templates = [template for template in team_template_dict.values()]
            # 直接比较模板ID，因为current_template现在就是原始模板字典
            return next((i for i, template in enumerate(templates) if template.get("id") == st.session_state.agent_chat_current_template.get("id")), 0)
        
        st.session_state.agent_chat_team_template_index = get_team_template_index()
        team_template = st.selectbox(
            i18n("Select a team template"),
            options=[template for template in team_template_manager.agent_templates.values()],
            format_func=lambda x: x.get("name"),
            index=st.session_state.agent_chat_team_template_index,
            key="agent_chat_team_template",
            on_change=team_template_change_callback
        )
        st.write(team_template.get("id"))
        st.write(st.session_state.agent_chat_current_template)

st.title(st.session_state.agent_chat_run_name)
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
