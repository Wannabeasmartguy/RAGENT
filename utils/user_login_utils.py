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

_DEFAULT_COOKIE = {
    'expiry_days': 30,
    'key': generate_random_string(64),
    'name': f'session_id_{generate_random_string(16)}'
}


def generate_secrets_yaml(credentials=_DEFAULT_USER_CREDENTIALS, cookie=_DEFAULT_COOKIE):
    """
    仅在 secrets.yaml 文件不存在时生成新的配置文件

    :param credentials: Dictionary containing user credentials.
    :param cookie: Dictionary containing cookie settings.
    """
    secrets_path = './secrets/secrets.yaml'
    
    # 如果配置文件已存在，直接返回
    if os.path.exists(secrets_path):
        return
        
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

    Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )
    return authenticator


if __name__ == '__main__':
    generate_secrets_yaml()