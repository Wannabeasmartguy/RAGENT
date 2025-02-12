import os
import re
import copy
import base64
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union, Tuple, Literal
from io import BytesIO
from functools import lru_cache

from core.llm._client_info import (
    OpenAISupportedClients,
)
from core.llm.ollama.completion import get_ollama_model_list
from core.llm.groq.completion import get_groq_models
from core.basic_config import I18nAuto
from core.processors import (
    OAILikeConfigProcessor,
    ChatProcessor,
    ALLDIAGLOGPROCESSOR
)
from config.constants import (
    I18N_DIR, 
    SUPPORTED_LANGUAGES,
    SUMMARY_PROMPT,
)
from utils.log.logger_config import setup_logger

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)


i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES["简体中文"]
)

@lru_cache(maxsize=10)
def model_selector(model_type):
    if model_type == OpenAISupportedClients.OPENAI.value:
        from openai import OpenAI
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_raw_list = client.models.list().data
            model_list = [model.id for model in model_raw_list]
            return model_list
        except Exception as e:
            logger.warning(f"Failed to get OpenAI model list: {e}")
            logger.info("Using default model list")
            return ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == OpenAISupportedClients.AOAI.value:
        from openai import AzureOpenAI
        try:
            client = AzureOpenAI(api_key=os.getenv("AZURE_OAI_KEY"), azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT"))
            model_list = client.models.list().data
            return [model.id for model in model_list]
        except Exception as e:
            logger.warning(f"Failed to get AOAI model list: {e}")
            logger.info("Using default model list")
            return ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == OpenAISupportedClients.OLLAMA.value:
        try:
            from openai import OpenAI
            client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="noneed")
            model_list = client.models.list().data
            return [model.id for model in model_list]
        except Exception as e:
            logger.warning(f"Failed to get Ollama model list: {e}")
            logger.info("Using request method to get model list")
            return get_ollama_model_list()
    elif model_type == OpenAISupportedClients.GROQ.value:
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_list = get_groq_models(api_key=groq_api_key,only_id=True)

            # exclude tts model
            model_list_exclude_tts = [model for model in model_list if "whisper" not in model]
            excluded_models = [model for model in model_list if model not in model_list_exclude_tts]

            logger.info(f"Groq model list: {model_list}, excluded models:{excluded_models}")
            return model_list_exclude_tts
        except Exception as e:
            logger.warning(f"Failed to get Groq model list: {e}")
            logger.info("Using default model list")
            return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    elif model_type == OpenAISupportedClients.OPENAI_LIKE.value:
        return ["Not given"]
    else:
        return None


def oai_model_config_selector(oai_model_config:Dict):
    config_processor = OAILikeConfigProcessor()
    model_name = list(oai_model_config.keys())[0]
    config_dict = config_processor.get_config()

    if model_name in config_dict:
        return model_name, config_dict[model_name]["base_url"], config_dict[model_name]["api_key"]
    else:
        return "Not given", "Not given", "Not given"


async def generate_new_run_name_with_llm_for_the_first_time(
    chat_history: List[Dict[str, Union[str, Dict, List]]],
    run_id: str,
    model_type: str,
    llm_config: Dict,
    dialog_processor: ALLDIAGLOGPROCESSOR,
    summary_prompt: str = SUMMARY_PROMPT,
) -> None:
    """根据对话内容，为首次进行对话的对话生成一个内容摘要的新名称"""
    import streamlit as st
    summary_chat_history = chat_history.copy()

    from utils.st_utils import generate_markdown_chat
    chat_history_md = generate_markdown_chat(
        chat_history=summary_chat_history
    )
    
    chat_processor = ChatProcessor(model_type=model_type, llm_config=llm_config)
    chat_history_summary = chat_processor.create_completion(
        messages=[
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": chat_history_md},
        ],
    )

    from modules.chat.transform import ReasoningContentTagProcessor
    tag_processor = ReasoningContentTagProcessor()
    _, new_run_name = tag_processor.extract(chat_history_summary.choices[0].message.content)
    dialog_processor.update_dialog_name(
        run_id=run_id,
        new_name=new_run_name,
    )

    st.rerun()


# def html_to_jpg(html_content: str) -> Image:
#     """
#     将HTML内容转换为JPG图片
#     """
#     image_bytes = html_to_image_bytes(html_content)
#     image = bytes_to_jpg(image_bytes)
#     return image

# def html_to_image_bytes(html_content: str) -> BytesIO:
#     """
#     将HTML内容转换为BytesIO对象
#     """
#     from weasyprint import HTML
    
#     html = HTML(string=html_content)
#     img_io = BytesIO()
#     html.write_png(img_io)
#     img_io.seek(0)
#     return img_io

# def bytes_to_jpg(bytes_content: BytesIO) -> Image:
#     """
#     将BytesIO内容转换为JPG图片
#     """
#     image = Image.open(bytes_content)
#     return image


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
    config_list = copy.deepcopy(config_list)
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


def current_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


def current_datetime_utc_str() -> str:
    return current_datetime_utc().strftime("%Y-%m-%dT%H:%M:%S")


def datetime_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def encode_image(image: BytesIO) -> str:
    """
    将 BytesIO 对象编码为 base64 字符串.
    
    Args:
        image (BytesIO): BytesIO 对象
    Returns:
        str: base64 编码的字符串
    """
    image_data = image.getvalue()
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return base64_encoded


def user_input_constructor(
    prompt: str, 
    images: Optional[Union[BytesIO, List[BytesIO]]] = None, 
) -> Dict:
    """
    构造用户多模态输入的字典。

    参数:
    - prompt: 用户输入的文本提示。
    - images: 可选的图像数据，可以是单个BytesIO对象或BytesIO对象的列表。

    返回值:
    - 一个字典，包含了用户的角色和内容（文本或/和图像）。
    """
    base_input = {
        "role": "user"
    }

    # 根据是否提供图像数据，构造不同的内容格式
    if images is None:
        # 如果没有图像，内容仅为文本提示
        base_input["content"] = prompt
    elif isinstance(images, (BytesIO, list)):
        # 如果有图像，内容为文本和图像的组合
        text_input = {
            "type": "text",
            "text": prompt
        }
        if isinstance(images, BytesIO):
            images = [images]
        
        image_inputs = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image)}"
                }
            } for image in images
        ]
        base_input["content"] = [text_input, *image_inputs]
    else:
        raise TypeError("images must be a BytesIO object or a list of BytesIO objects")

    return base_input