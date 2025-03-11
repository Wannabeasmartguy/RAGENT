from typing import List, Dict, Generator, Union, Optional, Literal
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import random
import time

from core.processors.chat.base import LoadBalanceStrategy
from utils.log.logger_config import setup_logger, get_load_balance_logger

from loguru import logger



class BaseLLM(ABC):
    def __init__(
        self,
        configs: Union[Dict, List[Dict]],
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    ):
        self.configs = [configs] if isinstance(configs, dict) else configs
        self.current_config_index = 0
        self.load_balance_strategy = load_balance_strategy
        self.config_usage = {
            i: {
                "usage": 0,
                "last_used": 0,
                "total_response_time": 0,
                "avg_response_time": 0,
                "weight": 1,
                "errors": 0,
                "last_error": None
            }
            for i in range(len(self.configs))
        }
        self.logger = get_load_balance_logger(self.load_balance_strategy)

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
            self.logger.info(
                f"Selected config in {selection_time:.3f}s",
                config_index=self.configs.index(selected_config)
            )
            return selected_config
            
        except Exception as e:
            self.logger.error(f"Error in config selection: {str(e)}")
            return self._round_robin_select()

    def _round_robin_select(self) -> Dict:
        """简单轮询策略"""
        selected_index = self.current_config_index
        self.current_config_index = (self.current_config_index + 1) % len(self.configs)
        
        logger.debug(
            f"Round Robin selected index {selected_index} "
            f"(next will be {self.current_config_index})"
        )
        self._update_usage_stats(selected_index)
        return self.configs[selected_index]

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
        return self.configs[selected_index]

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
                return self.configs[idx]
        
        logger.warning("Weighted selection failed, using first config")
        self._update_usage_stats(0)
        return self.configs[0]

    def _random_select(self) -> Dict:
        """随机策略"""
        selected_index = random.randint(0, len(self.configs) - 1)
        logger.debug(f"Random selected index {selected_index}")
        self._update_usage_stats(selected_index)
        return self.configs[selected_index]

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
        return self.configs[selected_index]

    def _update_usage_stats(self, index: int) -> None:
        """更新使用统计信息"""
        prev_usage = self.config_usage[index]["usage"]
        self.config_usage[index]["usage"] += 1
        self.config_usage[index]["last_used"] = time.time()
        
        self.logger.debug(
            f"Usage: {prev_usage} -> {self.config_usage[index]['usage']}",
            config_index=index
        )

    def update_response_time(self, index: int, response_time: float) -> None:
        """更新响应时间统计"""
        stats = self.config_usage[index]
        prev_avg = stats["avg_response_time"]
        
        stats["total_response_time"] += response_time
        stats["avg_response_time"] = stats["total_response_time"] / stats["usage"]
        
        self.logger.info(
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
        
        self.logger.warning(
            f"Errors: {prev_errors} -> {self.config_usage[index]['errors']}, "
            f"Weight: {prev_weight:.2f} -> {self.config_usage[index]['weight']:.2f}",
            config_index=index
        )

    def get_usage_stats(self) -> Dict:
        """获取使用统计信息"""
        for idx, stats in self.config_usage.items():
            self.logger.info(
                f"Stats summary: "
                f"usage={stats['usage']}, "
                f"avg_time={stats['avg_response_time']:.3f}s, "
                f"errors={stats['errors']}, "
                f"weight={stats['weight']:.2f}",
                config_index=idx
            )
        return self.config_usage

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]], stream: bool = False) -> Union[Dict, Generator]:
        pass