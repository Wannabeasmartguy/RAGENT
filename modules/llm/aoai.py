import time
from typing import List, Dict, Generator, Union, Optional

from utils.log.logger_config import setup_logger
from modules.llm.base import BaseLLM, LoadBalanceStrategy

from openai import AzureOpenAI
from loguru import logger


class AzureOpenAILLM(BaseLLM):
    def __init__(
        self,
        configs: Union[Dict, List[Dict]],
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        **params
    ):
        super().__init__(configs, load_balance_strategy)
        self.clients = []
        self.params = params
        for config in self.configs:
            self.clients.append(
                AzureOpenAI(
                    api_key=config.get("api_key"),
                    azure_endpoint=config.get("base_url"),
                    api_version=config.get("api_version")
                )
            )

    def invoke(
        self, messages: List[Dict[str, str]], stream: bool = False
    ) -> Dict | Generator:
        start_time = time.time()
        config = self._get_next_config()
        config_index = self.configs.index(config)
        
        try:
            response = self.clients[config_index].chat.completions.create(
                model=config["model"].replace(".", ""),
                messages=messages,
                stream=stream,
                **self.params
            )
            
            # 更新响应时间统计
            response_time = time.time() - start_time
            self.update_response_time(config_index, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Error with config {config_index}: {str(e)}")
            self.update_error_stats(config_index)
            
            if len(self.configs) > 1:
                logger.warning("Trying next config...")
                return self.invoke(messages, stream)
                
            raise ValueError(f"Error creating completion: {str(e)}") from e