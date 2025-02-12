import os
from dotenv import load_dotenv

from core.basic_config import I18nAuto
from config.constants import VERSION
from utils.log.logger_config import setup_logger
from utils.st_utils import set_pages_configs_in_common
from utils.user_login_utils import load_and_create_authenticator

import streamlit as st
from loguru import logger


load_dotenv(override=True)

try:
    set_pages_configs_in_common(
        title = "RAGENT",
        version = VERSION,
        init_sidebar_state = "expanded",
        layout="wide",
    )
except:
    st.rerun()


if os.getenv("LOGIN_ENABLED") == "True":
    from utils.user_login_utils import generate_secrets_yaml
    if not os.path.exists('./secrets'):
        generate_secrets_yaml()

    authenticator = load_and_create_authenticator()
    try:
        authenticator.login(
            fields={
                'Form name':'用户登录', 
                'Username':'用户名', 
                'Password':'密码',
                'Login':'登录', 
                'Captcha':'Captcha'
            }
        )
    except:
        st.error("登录失败")

    if st.session_state['authentication_status']:
        logger.debug("Login success, redirect to Classic Chat")
        authenticator.logout(location='sidebar')
        st.switch_page("pages/Classic_Chat.py")
    elif st.session_state['authentication_status'] is False:
        st.error('用户名或密码错误')
        logger.debug("Login failed, reason: Username or password error")
    elif st.session_state['authentication_status'] is None:
        st.warning('请输入用户名或密码')
        logger.debug("Login failed, reason: User did not enter username or password")

else:
    st.switch_page("pages/Classic_Chat.py")