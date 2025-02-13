import os
import yaml
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import Hasher
import secrets
import string

_DEFAULT_USER_CREDENTIALS = {
    'ragent': {
        'email': 'ragent@gmail.com',
        'failed_login_attempts': 0,
        'logged_in': False,
        'name': 'ragent',
        'password': 'test_pwd.123',
        'role': 'admin'
    },
}

def generate_random_string(length=32):
    """
    生成指定长度的随机字符串
    """
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def load_cookie_config():
    """
    从配置文件加载cookie配置，如果不存在则创建默认配置
    """
    secrets_path = './secrets/secrets.yaml'
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.SafeLoader)
            return config['cookie']
    
    # 如果配置不存在，创建新的配置
    cookie_config = {
        'expiry_days': 30,
        'key': generate_random_string(64),
        'name': f'session_id_{generate_random_string(16)}'
    }
    return cookie_config

_DEFAULT_COOKIE = load_cookie_config()

def generate_secrets_yaml(credentials=_DEFAULT_USER_CREDENTIALS, cookie=None):
    """
    仅在 secrets.yaml 文件不存在时生成新的配置文件

    :param credentials: Dictionary containing user credentials.
    :param cookie: Dictionary containing cookie settings.
    """
    secrets_path = './secrets/secrets.yaml'
    
    # 如果配置文件已存在，直接返回
    if os.path.exists(secrets_path):
        return
        
    # 如果没有传入cookie配置，使用默认配置
    if cookie is None:
        cookie = _DEFAULT_COOKIE
        
    data = {
        'credentials': {
            'usernames': credentials
        },
        'cookie': cookie,
    }

    if not os.path.exists('./secrets'):
        os.makedirs('./secrets')
    with open(secrets_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def load_and_create_authenticator():
    """
    Load and create an authenticator instance.

    :return: Authenticator instance.
    """
    with open('./secrets/secrets.yaml', 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    hashed_passwords = Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        credentials=hashed_passwords,
        cookie_name=config['cookie']['name'],
        cookie_key=config['cookie']['key'],
        cookie_expiry_days=config['cookie']['expiry_days'],
    )
    print(authenticator.cookie_controller.get_cookie())
    return authenticator


if __name__ == '__main__':
    generate_secrets_yaml()