import streamlit as st
import os

from configs.basic_config import I18nAuto, set_pages_configs_in_common, SUPPORTED_LANGUAGES
from configs.chat_config import CozeChatProcessor
from configs.pydantic_model import Bot_Single_Agent

VERSION = "0.0.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
set_pages_configs_in_common(version=VERSION,title="Coze-Agent",page_icon_path=logo_path)

# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])


with st.sidebar:
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

    with st.expander(label=i18n("Configration"), expanded=False):
        access_token = st.text_input(
            label=i18n("Access token"),
            key="access_token",
            help=i18n("(Optional)Input your Coze access token here"),
            placeholder=i18n("Optional")
        )
