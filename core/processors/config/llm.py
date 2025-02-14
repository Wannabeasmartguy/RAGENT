import os
import json
import uuid
from typing import Dict, List, Optional
from config.constants import OPENAI_LIKE_MODEL_CONFIG_FILE_PATH
from core.strategy import OpenAILikeModelConfigProcessStrategy, EncryptorStrategy
from core.encryption import FernetEncryptor
from core.models.llm import OpenAILikeConfigInStorage
from core.storage.oai_config import OpenAIConfigSQLiteStorage


class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """

    def __init__(self, encryptor: EncryptorStrategy = None):
        """
        Args:
            encryptor (EncryptorStrategy, optional): Defaults to None. If not provided, a new FernetEncryptor will be created.
        """
        self.encryptor = encryptor or FernetEncryptor()
        self.storage = OpenAIConfigSQLiteStorage()
    
    def reinitialize(self) -> None:
        """
        重新初始化类实例
        """
        self.__init__()   

    def add_model_config(
        self,
        user_id: str,
        model: str,
        base_url: str,
        api_key: str,
        custom_name: str = "",
        description: str = ""
    ) -> str:
        """
        添加模型的配置信息
        """
        config = OpenAILikeConfigInStorage(
            user_id=user_id,
            model=model,
            base_url=base_url,
            api_key=api_key,
            custom_name=custom_name,
            description=description
        )
        return self.storage.save_config(config)

    def delete_model_config(self, user_id: str, config_id: str) -> None:
        """
        删除模型的配置信息
        """
        self.storage.delete_config(user_id, config_id)

    def get_model_config(
        self,
        user_id: str,
        config_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict:
        """
        获取指定模型或配置ID的配置信息
        
        Args:
            user_id (str): 用户ID
            config_id (str, optional): 配置ID
            model (str, optional): 模型名称

        Returns:
            Dict: 匹配的配置信息字典
        """
        if config_id:
            config = self.storage.get_config(user_id, config_id)
            return config.model_dump() if config else {}
        
        if model:
            return {
                c.config_id: c.model_dump()
                for c in self.storage.list_configs(user_id)
                if c.model == model
            }
        return {}

    def list_model_configs(self, user_id: str) -> List[Dict]:
        """
        列出所有模型配置
        
        Args:
            user_id (str): 用户ID

        Returns:
            List[Dict]: 包含所有配置信息的列表
        """
        configs = self.storage.list_configs(user_id)
        return [{
            "id": c.config_id,
            "model": c.model,
            "custom_name": c.custom_name,
            "description": c.description,
            "base_url": "******",
            "api_key": "******"
        } for c in configs]
