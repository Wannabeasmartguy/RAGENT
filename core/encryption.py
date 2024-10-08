import os
from cryptography.fernet import Fernet
import base64

class Encryptor:
    def __init__(self, key=None):
        if key is None:
            key = os.getenv("ENCRYPTION_KEY")
            if not key:
                # 先检查是否存在 encryption_key.txt 文件
                if os.path.exists("encryption_key.txt"):
                    # 如果存在，则从文件中读取密钥
                    with open("encryption_key.txt", "r") as f:
                        key = f.read().encode()
                else:
                    # 如果不存在，则生成新的密钥
                    key = Fernet.generate_key()
                    print("警告: 未设置 `ENCRYPTION_KEY` 环境变量，已生成新的密钥。")
                    # 将密钥保存到文件中
                    with open("encryption_key.txt", "w") as f:
                        f.write(key.decode())
        self.cipher_suite = Fernet(key)

    def encrypt(self, data: str) -> str:
        return base64.urlsafe_b64encode(self.cipher_suite.encrypt(data.encode())).decode()

    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher_suite.decrypt(base64.urlsafe_b64decode(encrypted_data)).decode()
