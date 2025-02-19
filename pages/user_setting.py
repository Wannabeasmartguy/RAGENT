import os

from utils.st_utils import set_pages_configs_in_common
from config.constants import (
    VERSION, 
    LOGO_DIR, 
    I18N_DIR, 
    SUPPORTED_LANGUAGES
)
from utils.st_utils import (
    keep_login_or_logout_and_redirect_to_login_page,
    reset_user_password
)
from utils.user_login_utils import load_and_create_authenticator
from core.basic_config import I18nAuto

import streamlit as st

language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(i18n_dir=I18N_DIR, language=SUPPORTED_LANGUAGES[language])

logo_path = os.path.join(LOGO_DIR, "RAGENT_logo.png")
logo_text_path = os.path.join(LOGO_DIR, "RAGENT_logo_with_text_horizon.png")
try:
    set_pages_configs_in_common(
        version=VERSION,
        title="RAGENT",
        page_icon_path=logo_path,
    )
except:
    st.rerun()

authenticator = load_and_create_authenticator()
keep_login_or_logout_and_redirect_to_login_page(
    authenticator=authenticator,
    logout_key="user_setting_logout",
    login_page="RAGENT.py"
)

with st.sidebar:
    st.logo(logo_text_path)

    st.page_link("pages/Classic_Chat.py", label="💭 Classic Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 Agent Chat")

st.title(i18n("User Setting"))
st.subheader(i18n("User Info"))
st.write(f"Hello, {st.session_state['name']}!")
st.write(f"Your email is {st.session_state['email']}.")

st.subheader(i18n("Reset Password"))
reset_user_password(authenticator)

