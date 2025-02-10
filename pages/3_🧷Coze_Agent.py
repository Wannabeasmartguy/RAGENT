import streamlit as st
import os
import uuid

from core.basic_config import I18nAuto, set_pages_configs_in_common
from core.processors.chat.classic import CozeChatProcessor
from core.storage.displayer.coze import display_cozebot_response, display_coze_conversation
from config.constants.i18n import I18N_DIR, SUPPORTED_LANGUAGES

VERSION = "0.1.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'assets', 'images', 'logos', 'RAGENT_logo.png')
logo_text = os.path.join(parent_directory, "assets", "images", "logos", "RAGENT_logo_with_text_horizon.png")
set_pages_configs_in_common(version=VERSION,title="Coze-Agent",page_icon_path=logo_path)

language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES[language]
)

if "coze_chat_history_display" not in st.session_state:
    st.session_state.coze_chat_history_display = []

if "coze_chat_history_total" not in st.session_state:
    st.session_state.coze_chat_history_total = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

display_coze_conversation(st.session_state.coze_chat_history_display)
# st.write(st.session_state.coze_chat_history_total)

with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("RAGENT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

    "---"

    st.page_link("pages/Coze_Bot_info.py", label="ℹ️ Coze Bot")

    with st.expander(label=i18n("Configration"), expanded=False):
        access_token = st.text_input(
            label=i18n("Access token"),
            value=os.getenv("COZE_ACCESS_TOKEN",None),
            key="access_token",
            help=i18n("(Optional)Input your Coze access token here"),
            placeholder=i18n("Optional")
        )
        bot_id = st.text_input(
            label=i18n("Bot ID"),
            key="bot_id",
            help=i18n("Your Coze bot ID, see https://www.coze.cn/docs/developer_guides/coze_api_overview#c5ac4993"),
        )
    
    cols = st.columns(2)
    export_button = cols[0].button(label=i18n("Export chat history"))
    clear_button = cols[1].button(label=i18n("Clear chat history"))

    if clear_button:
        st.session_state.coze_chat_history_display = []
        st.session_state.coze_chat_history_total = []
        st.rerun()


coze_chat_processor = CozeChatProcessor(access_token=access_token)

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.coze_chat_history_total.append({"role": "user", "content": prompt, "content_type":"text"})
    st.session_state.coze_chat_history_display.append({"role": "user", "content": prompt, "content_type":"text"})

    with st.chat_message("assistant"):
        with st.spinner(text=i18n("Thinking...")):
            response = coze_chat_processor.create_coze_agent_response(
                user='user',
                query=prompt,
                bot_id=bot_id,
                conversation_id=st.session_state.conversation_id,
                chat_history=st.session_state.coze_chat_history_total,
            )
        usable_response = display_cozebot_response(response.json())

        st.session_state.coze_chat_history_display.append({"role": "assistant", "content": usable_response, "content_type":"coze_bot_response"})
        for i in usable_response.messages:
            if i.type != 'follow_up':
                st.session_state.coze_chat_history_total.append(dict(i))