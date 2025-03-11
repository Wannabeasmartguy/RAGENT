from enum import Enum

class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"  # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接数
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    RANDOM = "random"  # 随机
    LEAST_RESPONSE_TIME = "least_response_time"  # 最短响应时间