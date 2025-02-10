import os
import json
import uuid
from typing import Dict, List, Optional
from config.constants import OPENAI_LIKE_MODEL_CONFIG_FILE_PATH
from core.strategy import OpenAILikeModelConfigProcessStrategy, EncryptorStrategy
from core.encryption import FernetEncryptor


class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """
    config_path = OPENAI_LIKE_MODEL_CONFIG_FILE_PATH

    def __init__(self, encryptor: EncryptorStrategy = None):
        """
        Args:
            encryptor (EncryptorStrategy, optional): Defaults to None. If not provided, a new FernetEncryptor will be created.
        """
        self.encryptor = encryptor or FernetEncryptor()
        # 如果本地没有custom_model_config.json文件，则创建文件夹及文件
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump({}, f, indent=4)
        
        # 读取custom_model_config.json文件
        with open(self.config_path, "r") as f:
            self.exist_config = json.load(f)
    
    def reinitialize(self) -> None:
        """
        重新初始化类实例
        """
        self.__init__()

    def get_config(self) -> Dict:
        """
        获取完整的配置文件配置信息
        """
        return self.exist_config
    
    def update_config(
            self, 
            model: str,
            base_url: str,
            api_key: str,
            custom_name: str = "",
            description: str = "",
        ) -> str:
        """
        更新或添加模型的配置信息
        
        Args:
            model (str): 模型名称
            base_url (str): API基础URL
            api_key (str): API密钥
            description (str): 配置描述，用于区分相同模型的不同配置

        Returns:
            str: 配置的唯一标识符
        """
        config_id = str(uuid.uuid4())
        config = {
            "model": model,
            "base_url": self.encryptor.encrypt(base_url),
            "api_key": self.encryptor.encrypt(api_key),
            "custom_name": custom_name,
            "description": description
        }
        self.exist_config[config_id] = config
        
        # 更新custom_model_config.json文件
        with open(self.config_path, "w") as f:
            json.dump(self.exist_config, f, indent=4)
        
        return config_id
        
    def delete_model_config(self, config_id: str) -> None:
        """
        删除模型的配置信息
        """
        if config_id in self.exist_config:
            del self.exist_config[config_id]
            
            # 更新custom_model_config.json文件
            with open(self.config_path, "w") as f:
                json.dump(self.exist_config, f, indent=4)
                
    def get_model_config(self, model: str = None, config_id: str = None) -> Dict:
        """
        获取指定模型或配置ID的配置信息
        
        Args:
            model (str, optional): 模型名称
            config_id (str, optional): 配置ID

        Returns:
            Dict: 匹配的配置信息字典
        """
        if config_id:
            config = self.exist_config.get(config_id, {})
            if config:
                config["base_url"] = self.encryptor.decrypt(config["base_url"])
                config["api_key"] = self.encryptor.decrypt(config["api_key"])
            return config
        elif model:
            return {
                config_id: {
                    **config,
                    "base_url": self.encryptor.decrypt(config["base_url"]),
                    "api_key": self.encryptor.decrypt(config["api_key"]),
                }
                for config_id, config in self.exist_config.items()
                if config["model"] == model
            }
        else:
            return {}

    def list_model_configs(self) -> List[Dict]:
        """
        列出所有模型配置
        
        Returns:
            List[Dict]: 包含所有配置信息的列表
        """
        return [
            {
                "id": config_id, 
                **{k: v if k not in ['base_url', 'api_key'] else '******' for k, v in config.items()}
            }
            for config_id, config in self.exist_config.items()
        ]
