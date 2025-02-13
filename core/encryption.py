import os
import base64
from cryptography.fernet import Fernet
from loguru import logger

from core.strategy import EncryptorStrategy


class FernetEncryptor(EncryptorStrategy):
    def __init__(self, key=None):
        if key is None:
            key = os.getenv("ENCRYPTION_KEY")
            if key:
                try:
                    # 尝试使用环境变量中的密钥
                    Fernet(key.encode() if isinstance(key, str) else key)
                except ValueError:
                    # 如果环境变量中的密钥不合法，生成新密钥
                    logger.warning(
                        "Invalid `ENCRYPTION_KEY` in environment variable, generating a new key."
                    )
                    key = None
            
            if not key:
                # 如果没有合法的环境变量密钥，尝试从文件读取或生成新密钥
                if os.path.exists("encryption_key.txt"):
                    with open("encryption_key.txt", "r") as f:
                        key = f.read().encode()
                else:
                    key = Fernet.generate_key()
                    logger.warning(
                        "Generating a new encryption key."
                    )
                # 将密钥保存到文件中（无论是读取还是新生成的）
                with open("encryption_key.txt", "w") as f:
                    f.write(key.decode() if isinstance(key, bytes) else key)
        
        # 确保最终的key是bytes类型
        if isinstance(key, str):
            key = key.encode()
        self.cipher_suite = Fernet(key)

    def encrypt(self, data: str) -> str:
        return base64.urlsafe_b64encode(
            self.cipher_suite.encrypt(data.encode())
        ).decode()

    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher_suite.decrypt(
            base64.urlsafe_b64decode(encrypted_data)
        ).decode()
