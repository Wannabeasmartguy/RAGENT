from typing import Literal, List, Dict, Any, Generator, Optional, Union
from functools import partial
from uuid import uuid4
from deprecated import deprecated
import os
import json
import uuid
import requests
import time
import random
from datetime import datetime, timedelta

from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion

from core.processors.chat.base import LoadBalanceStrategy
from core.llm._client_info import SUPPORTED_SOURCES as SUPPORTED_CLIENTS
from core.llm._client_info import (
    OpenAISupportedClients,
    validate_client_config
)
from core.strategy import (
    ChatProcessStrategy,
    OpenAILikeModelConfigProcessStrategy,
    CozeChatProcessStrategy
)
from core.strategy import EncryptorStrategy
from core.encryption import FernetEncryptor
from utils.tool_utils import create_tools_call_completion
from utils.log.logger_config import setup_logger, get_load_balance_logger


class ChatProcessor(ChatProcessStrategy):
    """
    处理聊天消息的策略模式实现类
    """
    def __init__(
            self, 
            model_type: str,
            llm_config: Union[Dict, List[Dict]],
            load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
        ) -> None:
        self.model_type = model_type
        self.llm_configs = [llm_config] if isinstance(llm_config, dict) else llm_config
        self.llm_configs = [validate_client_config(model_type, config) for config in self.llm_configs]
        self.create_tools_call_completion = partial(create_tools_call_completion, config_list=self.llm_configs)
        
        self.current_config_index = 0
        self.load_balance_strategy = load_balance_strategy
        self.config_usage = {
            i: {
                "usage": 0,  # 使用次数
                "last_used": 0,  # 最后使用时间
                "total_response_time": 0,  # 总响应时间
                "avg_response_time": 0,  # 平均响应时间
                "weight": 1,  # 权重，默认为1
                "errors": 0,  # 错误次数
                "last_error": None  # 最后错误时间
            } 
            for i in range(len(self.llm_configs))
        }
        
        self.lb_logger = get_load_balance_logger(load_balance_strategy.value)
        self.lb_logger.info(
            f"Initializing ChatProcessor with {len(self.llm_configs)} configurations"
        )
        
        # 记录每个配置的基本信息
        for i, config in enumerate(self.llm_configs):
            logger.debug(
                f"Config {i}: model={config.get('model')}, "
                f"base_url={config.get('base_url', '******')}"
            )
    
    def _get_next_config(self) -> Dict:
        """根据选择的负载均衡策略获取下一个配置"""
        start_time = time.time()
        selected_config = None
        
        try:
            if self.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
                selected_config = self._round_robin_select()
            elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                selected_config = self._least_connections_select()
            elif self.load_balance_strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                selected_config = self._weighted_round_robin_select()
            elif self.load_balance_strategy == LoadBalanceStrategy.RANDOM:
                selected_config = self._random_select()
            elif self.load_balance_strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
                selected_config = self._least_response_time_select()
            else:
                selected_config = self._round_robin_select()
                
            selection_time = time.time() - start_time
            self.lb_logger.info(
                f"Selected config in {selection_time:.3f}s",
                config_index=self.llm_configs.index(selected_config)
            )
            return selected_config
            
        except Exception as e:
            self.lb_logger.error(f"Error in config selection: {str(e)}")
            return self._round_robin_select()

    def _round_robin_select(self) -> Dict:
        """简单轮询策略"""
        selected_index = self.current_config_index
        self.current_config_index = (self.current_config_index + 1) % len(self.llm_configs)
        
        logger.debug(
            f"Round Robin selected index {selected_index} "
            f"(next will be {self.current_config_index})"
        )
        self._update_usage_stats(selected_index)
        return self.llm_configs[selected_index]

    def _least_connections_select(self) -> Dict:
        """最少连接数策略"""
        min_usage = float('inf')
        selected_index = 0
        
        for idx, stats in self.config_usage.items():
            if stats["usage"] < min_usage:
                min_usage = stats["usage"]
                selected_index = idx
        
        logger.debug(
            f"Least Connections selected index {selected_index} "
            f"(usage={min_usage})"
        )
        self._update_usage_stats(selected_index)
        return self.llm_configs[selected_index]

    def _weighted_round_robin_select(self) -> Dict:
        """加权轮询策略"""
        total_weight = sum(stats["weight"] for stats in self.config_usage.values())
        target = random.uniform(0, total_weight)
        current_weight = 0
        
        for idx, stats in self.config_usage.items():
            current_weight += stats["weight"]
            if current_weight >= target:
                logger.debug(
                    f"Weighted Round Robin selected index {idx} "
                    f"(weight={stats['weight']:.2f}, target={target:.2f})"
                )
                self._update_usage_stats(idx)
                return self.llm_configs[idx]
        
        logger.warning("Weighted selection failed, using first config")
        self._update_usage_stats(0)
        return self.llm_configs[0]

    def _random_select(self) -> Dict:
        """随机策略"""
        selected_index = random.randint(0, len(self.llm_configs) - 1)
        logger.debug(f"Random selected index {selected_index}")
        self._update_usage_stats(selected_index)
        return self.llm_configs[selected_index]

    def _least_response_time_select(self) -> Dict:
        """最短响应时间策略"""
        min_response_time = float('inf')
        selected_index = 0
        
        for idx, stats in self.config_usage.items():
            if stats["avg_response_time"] < min_response_time:
                min_response_time = stats["avg_response_time"]
                selected_index = idx
        
        logger.debug(
            f"Least Response Time selected index {selected_index} "
            f"(avg_time={min_response_time:.3f}s)"
        )
        self._update_usage_stats(selected_index)
        return self.llm_configs[selected_index]

    def _update_usage_stats(self, index: int) -> None:
        """更新使用统计信息"""
        prev_usage = self.config_usage[index]["usage"]
        self.config_usage[index]["usage"] += 1
        self.config_usage[index]["last_used"] = time.time()
        
        self.lb_logger.debug(
            f"Usage: {prev_usage} -> {self.config_usage[index]['usage']}",
            config_index=index
        )

    def update_response_time(self, index: int, response_time: float) -> None:
        """更新响应时间统计"""
        stats = self.config_usage[index]
        prev_avg = stats["avg_response_time"]
        
        stats["total_response_time"] += response_time
        stats["avg_response_time"] = stats["total_response_time"] / stats["usage"]
        
        self.lb_logger.info(
            f"Response time: current={response_time:.3f}s, "
            f"avg={prev_avg:.3f}s -> {stats['avg_response_time']:.3f}s",
            config_index=index
        )

    def update_error_stats(self, index: int) -> None:
        """更新错误统计"""
        prev_errors = self.config_usage[index]["errors"]
        prev_weight = self.config_usage[index]["weight"]
        
        self.config_usage[index]["errors"] += 1
        self.config_usage[index]["last_error"] = datetime.now()
        
        error_penalty = min(0.2 * self.config_usage[index]["errors"], 0.8)
        self.config_usage[index]["weight"] = max(0.2, 1 - error_penalty)
        
        self.lb_logger.warning(
            f"Errors: {prev_errors} -> {self.config_usage[index]['errors']}, "
            f"Weight: {prev_weight:.2f} -> {self.config_usage[index]['weight']:.2f}",
            config_index=index
        )

    def get_usage_stats(self) -> Dict:
        """获取使用统计信息"""
        for idx, stats in self.config_usage.items():
            self.lb_logger.info(
                f"Stats summary: "
                f"usage={stats['usage']}, "
                f"avg_time={stats['avg_response_time']:.3f}s, "
                f"errors={stats['errors']}, "
                f"weight={stats['weight']:.2f}",
                config_index=idx
            )
        return self.config_usage

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> ChatCompletion | Generator:
        start_time = time.time()
        source = self.model_type.lower()
        
        if source not in SUPPORTED_CLIENTS:
            logger.error(f"Unsupported source: {source}")
            raise ValueError(f"Unsupported source: {source}")
        
        if any(client.value == source for client in OpenAISupportedClients):
            current_config = self._get_next_config()
            config_index = self.llm_configs.index(current_config)
            
            try:
                if current_config.get("api_type") != "azure":
                    from openai import OpenAI
                    client = OpenAI(
                        api_key=current_config.get("api_key"),
                        base_url=current_config.get("base_url"),
                    )
                else:
                    from openai import AzureOpenAI
                    client = AzureOpenAI(
                        api_key=current_config.get("api_key"),
                        azure_endpoint=current_config.get("base_url"),
                        api_version=current_config.get("api_version"),
                    )
                    
                params = {
                    "model": current_config.get("model").replace(".", "") if current_config.get("api_type") == "azure" else current_config.get("model"),
                    "messages": messages,
                    "temperature": current_config.get("params", {}).get("temperature", 0.5),
                    "top_p": current_config.get("params", {}).get("top_p", 0.1),
                    "max_tokens": current_config.get("params", {}).get("max_tokens", 4096),
                    "stream": stream
                }
                
                response = client.chat.completions.create(**params)
                
                # 更新响应时间统计
                response_time = time.time() - start_time
                self.update_response_time(config_index, response_time)
                
                return response
                
            except Exception as e:
                logger.error(f"Error with config {config_index}: {str(e)}")
                self.update_error_stats(config_index)
                
                if len(self.llm_configs) > 1:
                    logger.warning("Trying next config...")
                    return self.create_completion(messages, stream)
                    
                raise ValueError(f"Error creating completion: {str(e)}") from e


@deprecated("OAILikeConfigProcessor is deprecated. Use OAILikeConfigProcessor in core.processors.config.llm instead.")
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
        

class CozeChatProcessor(CozeChatProcessStrategy):
    """
    处理与 Coze API 相关的逻辑
    """
    def __init__(
            self, 
            access_token: Optional[str] = None,
        ):
        
        if access_token:
            self.personal_access_token = access_token
        else:
            from dotenv import load_dotenv
            load_dotenv()
            self.personal_access_token = os.getenv("COZE_ACCESS_TOKEN")
    
    def create_coze_agent_response(
            self,
            user: str,
            query: str,
            bot_id: str,
            stream: bool = False,
            conversation_id: Optional[str] = None,
            chat_history: Optional[List[Dict[str, str]]] = None,
            custom_variables: Optional[Dict[str, str]] = None,
        ) -> requests.Response:
        """
        Creates a response from the Coze Agent API.

        Args:
            user (str): The user identifier.
            query (str): The user's input or question.
            bot_id (str): Identifier for the bot to use.
            stream (bool, optional): Whether to return streaming responses. Defaults to False.
            conversation_id (Optional[str], optional): ID of the current conversation. Defaults to None.
            chat_history (Optional[List[Dict[str, str]]], optional): History of previous interactions. Defaults to None.
            custom_variables (Optional[Dict[str, str]], optional): Custom variables for the request. Defaults to None.

        Returns:
            requests.Response: The response from the API call.
        """
        headers = {
            'Authorization': f'Bearer {self.personal_access_token}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': 'api.coze.cn',
            'Connection': 'keep-alive'
        }

        data = {
            'user': user,
            'query': query,
            'bot_id': bot_id,
            'stream': stream,
            'conversation_id': conversation_id,
            'chat_history': chat_history,
            'custom_variables': custom_variables
        }

        data = dict(filter(lambda item: item[1] is not None, data.items()))

        response = requests.post(
            url='https://api.coze.cn/open_api/v2/chat',
            headers=headers,
            json=data
        )

        return response

    @classmethod
    def get_bot_config(
            cls,
            personal_access_token: str,
            bot_id: str,
            bot_version: Optional[str] = None
        ) -> requests.Response:
        headers = {
            'Authorization': f'Bearer {personal_access_token}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Connection': 'keep-alive'
        }
        
        params = {
            'bot_id': bot_id,
            'bot_version': bot_version
        }
        params = dict(filter(lambda item: item[1] is not None, params.items()))

        response = requests.get(
            url="https://api.coze.cn/v1/bot/get_online_info",
            headers=headers,
            params=params
        )
        return response