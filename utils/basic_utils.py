import streamlit as st
from streamlit import cache_resource

import os
import json
import uuid
from typing import List, Dict

from llm.ollama.completion import get_ollama_model_list
from configs.chat_config import OAILikeConfigProcessor


def model_selector(model_type):
    if model_type == "OpenAI" or model_type == "AOAI":
        return ["gpt-3.5-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        try:
           model_list = get_ollama_model_list() 
           return model_list
        except:
            return ["qwen:7b-chat"]
    elif model_type == "Groq":
        return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    elif model_type == "Llamafile":
        return ["Noneed"]
    else:
        return None


def oai_model_config_selector(oai_model_config:Dict):
    config_processor = OAILikeConfigProcessor()
    model_name = list(oai_model_config.keys())[0]
    config_dict = config_processor.get_config()

    if model_name in config_dict:
        return model_name, config_dict[model_name]["base_url"], config_dict[model_name]["api_key"]
    else:
        return "noneed", "http://127.0.0.1:8080/v1", "noneed"


# Display chat messages from history on app rerun
@st.cache_data
def write_chat_history(chat_history: List[Dict[str, str]]) -> None:
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def split_list_by_key_value(dict_list, key, value):
    result = []
    temp_list = []
    count = 0

    for d in dict_list:
        # 检查字典是否有指定的key，并且该key的值是否等于指定的value
        if d.get(key) == value:
            count += 1
            temp_list.append(d)
            # 如果指定值的出现次数为2，则分割列表
            if count == 2:
                result.append(temp_list)
                temp_list = []
                count = 0
        else:
            # 如果当前字典的key的值不是指定的value，则直接添加到当前轮次的列表
            temp_list.append(d)

    # 将剩余的临时列表（如果有）添加到结果列表
    if temp_list:
        result.append(temp_list)

    return result


class Meta(type):
    def __new__(cls, name, bases, attrs):
        for name, value in attrs.items():
            if callable(value) and not name.startswith('__') and not name.startswith('_'):  # 跳过特殊方法和私有方法
                attrs[name] = cache_resource(value)
        return super().__new__(cls, name, bases, attrs)
    

def save_basic_chat_history(
        chat_name: str,
        chat_history: List[Dict[str, str]], 
        chat_history_file: str = 'chat_history.json'):
    """
    保存一般 LLM Chat 聊天记录
    
    Args:
        user_id (str): 用户id
        chat_history (List[Tuple[str, str]]): 聊天记录
        chat_history_file (str, optional): 聊天记录文件. Defaults to 'chat_history.json'.
    """
    # TODO: 添加重名检测，如果重名则添加时间戳
    # 如果聊天历史记录文件不存在，则创建一个空的文件
    if not os.path.exists(chat_history_file):
        with open(chat_history_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    
    # 打开聊天历史记录文件，读取数据
    with open(chat_history_file, 'r', encoding='utf-8') as f:
        data:dict = json.load(f)
        
    # 如果聊天室名字不在数据中，则添加聊天的名字和完整聊天历史记录
    if chat_name not in data:
        data.update(
            {
                chat_name: {
                    "chat_history":chat_history,
                    "id": str(uuid.uuid4())
                }
            }
        )
        
    # 打开聊天历史记录文件，写入更新后的数据
    with open(chat_history_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def list_length_transform(n, lst) -> List:
    '''
    聊天上下文限制函数
    
    Args:
        n (int): 限制列表lst的长度为n
        lst (list): 需要限制长度的列表
        
    Returns:
        list: 限制后的列表
    '''
    # 如果列表lst的长度大于n，则返回lst的最后n个元素
    if len(lst) > n:
        return lst[-n:]
    # 如果列表lst的长度小于等于n，则返回lst本身
    else:
        return lst


def detect_and_decode(data_bytes):
    """
    尝试使用不同的编码格式来解码bytes对象。

    参数:
    data_bytes (bytes): 需要进行编码检测和解码的bytes对象。

    返回:
    tuple: 包含解码后的字符串和使用的编码格式。
           如果解码失败，返回错误信息。
    """
    # 定义常见的编码格式列表
    encodings = ['utf-8', 'ascii', 'gbk', 'iso-8859-1']

    # 遍历编码格式，尝试解码
    for encoding in encodings:
        try:
            # 尝试使用当前编码解码
            decoded_data = data_bytes.decode(encoding)
            # 如果解码成功，返回解码后的数据和编码格式
            return decoded_data, encoding
        except UnicodeDecodeError:
            # 如果当前编码解码失败，继续尝试下一个编码
            continue

    # 如果所有编码格式都解码失败，返回错误信息
    return "无法解码，未知的编码格式。", None


def config_list_postprocess(config_list: List[Dict]):
    """将config_list中，每个config的params字段合并到各个config中。"""
    for config in config_list:
        if "params" in config:
            params = config["params"]
            del config["params"]
            config.update(**params)
    return config_list


def dict_filter(
        dict_data: Dict, 
        filter_keys: List[str] = None,
        filter_values: List[str] = None
    ) -> Dict:
    """
    过滤字典中的键值对，只保留指定的键或值。
    
    Args:
        dict_data (Dict): 要过滤的字典。
        filter_keys (List[str], optional): 要保留的键列表。默认值为None。
        filter_values (List[str], optional): 要保留的值列表。默认值为None。
        
    Returns:
        Dict: 过滤后的字典。
    """
    if filter_keys is None and filter_values is None:
        return dict_data
    
    filtered_dict = {}
    for key, value in dict_data.items():
        if (filter_keys is None or key in filter_keys) and (filter_values is None or value in filter_values):
            filtered_dict[key] = value
            
    return filtered_dict


def reverse_traversal(lst: List) -> Dict[str, str]:
    '''
    反向遍历列表，直到找到非空且不为'TERMINATE'的元素为止。
    用于处理 Tool Use Agent 的 chat_history 列表。
    '''
    # 遍历列表中的每一个元素
    for item in reversed(lst):
        # 如果元素中的内容不为空且不为'TERMINATE'，则打印元素
        if item.get('content', '') not in ('', 'TERMINATE'):
            return item