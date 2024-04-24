"""用于测试展示输入输出的模块"""
from types import SimpleNamespace

def fake_agent_chat_completion(query:str):
    """
    提供agent chat的模拟输出
    
    Returm:
        List[dict]
    """
    response = SimpleNamespace()
    response.chat_history = [
        {
            "role": "user",
            "content": query,
            "if_thought": 0
        },
        {
            "role": "assistant",
            "content": "answer 2",
            "if_thought": 1
        },
        {
            "role": "user",
            "content": "answer 3",
            "if_thought": 1
        },
        {
            "role": "assistant",
            "content": "answer 4",
            "if_thought": 1
        },
        {
            "role": "user",
            "content": "answer 5",
            "if_thought": 0
        }
    ]
    response.summary = "answer 5"
    
    return response