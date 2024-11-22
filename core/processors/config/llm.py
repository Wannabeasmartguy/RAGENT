import os
import json
import uuid
from typing import Dict, List, Optional
from core.strategy import OpenAILikeModelConfigProcessStrategy, EncryptorStrategy
from core.encryption import FernetEncryptor
from core.basic_config import (
    CONFIGS_BASE_DIR,
    CONFIGS_DB_FILE,
    EMBEDDING_CONFIGS_DB_TABLE,
    LLM_CONFIGS_DB_TABLE
)
from storage.db.sqlite import (
    SqlAssistantLLMConfigStorage,
)
from model.config.llm import OpenAILikeLLMConfiguration


class OAILikeConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类
    """
    config_path = os.path.join("dynamic_configs", "custom_model_config.json")

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


class OAILikeSqliteConfigProcessor(OpenAILikeModelConfigProcessStrategy):
    """
    处理 OAI-like 模型的配置的策略模式实现类，数据保存在 SQlite 中
    """
    def __init__(
            self,
            table_name: str = LLM_CONFIGS_DB_TABLE,
            db_url: Optional[str] = None,
            db_file: Optional[str] = CONFIGS_DB_FILE
        ):
        self.storage = SqlAssistantLLMConfigStorage(
            table_name=table_name,
            db_url=db_url,
            db_file=db_file
        )
        self.models_table = self.storage.get_all_models()
    
    def reinitialize(self) -> None:
        """
        重新初始化类实例
        """
        self.__init__()

    def get_config(self) -> Dict:
        """
        获取完整的配置文件配置信息的字典
        """
        model_config = {}
        for model in self.models_table:
            model_info = model.dict()
            model_config[model.model_name] = model_info
        return model_config
    
    def update_config(
        self,
        model_name:str,
        api_key: str,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        更新模型的配置信息
        
        Args:
            model_name (str): 模型的名称
            api_key (str): 模型的API Key
            base_url (Optional[str], optional): 访问模型的端点.
            kwargs (Dict): 其他配置信息
        """
        oai_like_config = OpenAILikeLLMConfiguration(
            model_id=str(uuid.uuid4()),
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self.storage.upsert(oai_like_config)