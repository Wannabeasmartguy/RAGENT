import streamlit as st
import os

from core.basic_config import I18nAuto, set_pages_configs_in_common, SUPPORTED_LANGUAGES
from utils.coze_utils import display_bot_info


VERSION = "0.1.1"
current_directory = os.path.dirname(__file__)
parent_directory = os.path.dirname(current_directory)
logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
logo_text = os.path.join(parent_directory, "img", "RAGenT_logo_with_text_horizon.png")
set_pages_configs_in_common(version=VERSION,title="Coze Bot info",page_icon_path=logo_path)

language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])


with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("pages/3_🧷Coze_Agent.py", label="🧷 Coze Agent")

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
    get_info_button = st.button(
        label=i18n("Get Coze Bot info"), 
        on_click=display_bot_info,
        args=(access_token,bot_id),
        use_container_width=True
    )