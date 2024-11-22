import streamlit as st
import pyperclip
from streamlit import cache_resource
from pkg_resources import parse_version

import os
import re
import copy
import base64
import textwrap
from datetime import datetime, timezone
from loguru import logger
from functools import lru_cache
from typing import List, Dict, Optional, Union, Tuple, Literal
from io import BytesIO
from dotenv import load_dotenv
load_dotenv(override=True)

from llm.ollama.completion import get_ollama_model_list
from llm.groq.completion import get_groq_models
from core.basic_config import I18nAuto, SUPPORTED_LANGUAGES
from core.processors.chat.classic import OAILikeConfigProcessor
from css.export_themes import default_theme, glassmorphism_theme
from css.classic_chat_css import (
    USER_CHAT_STYLE_ST_V37,
    USER_CHAT_STYLE_ST_V39,
    ASSISTANT_CHAT_STYLE,
)
from css.rag_chat_css import (
    RAG_CHAT_USER_STYLE_ST_V37,
    RAG_CHAT_USER_STYLE_ST_V39,
    RAG_CHAT_ASSISTANT_STYLE_ST_V37,
    RAG_CHAT_ASSISTANT_STYLE_ST_V39,
)

USER_AVATAR_SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-user-square" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#1455ea" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
    <path d="M9 10a3 3 0 1 0 6 0a3 3 0 0 0 -6 0" />
    <path d="M6 21v-1a4 4 0 0 1 4 -4h4a4 4 0 0 1 4 4v1" />
    <path d="M3 5a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2v14a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2v-14z" />
    </svg>
"""

AI_AVATAR_SVG = """
    <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-message-chatbot" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="#1455ea" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
    <path d="M18 4a3 3 0 0 1 3 3v8a3 3 0 0 1 -3 3h-5l-5 3v-3h-2a3 3 0 0 1 -3 -3v-8a3 3 0 0 1 3 -3h12z" />
    <path d="M9.5 9h.01" />
    <path d="M14.5 9h.01" />
    <path d="M9.5 13a3.5 3.5 0 0 0 5 0" />
    </svg>
"""

# 定义样式常量
STYLE_CONSTANTS = {
    "USER_CHAT": {
        "v37": USER_CHAT_STYLE_ST_V37,
        "v39": USER_CHAT_STYLE_ST_V39,
    },
    "ASSISTANT_CHAT": {
        "v37": ASSISTANT_CHAT_STYLE,
        "v39": ASSISTANT_CHAT_STYLE,
    },
    "RAG_USER_CHAT": {
        "v37": RAG_CHAT_USER_STYLE_ST_V37,
        "v39": RAG_CHAT_USER_STYLE_ST_V39,
    },
    "RAG_ASSISTANT_CHAT": {
        "v37": RAG_CHAT_ASSISTANT_STYLE_ST_V37,
        "v39": RAG_CHAT_ASSISTANT_STYLE_ST_V39,
    }
}


def get_style(
        style_type:Literal["USER_CHAT", "ASSISTANT_CHAT", "RAG_USER_CHAT", "RAG_ASSISTANT_CHAT"],
        st_version:str
    ) -> str:
    """
    根据样式类型和Streamlit版本获取相应的样式。
    
    :param style_type: 样式类型，如 "USER_CHAT", "ASSISTANT_CHAT" 等
    :param st_version: Streamlit版本号, 形如 "1.37.0"
    :return: 对应的样式字符串
    """
    version = parse_version(st_version)
    style_dict = STYLE_CONSTANTS.get(style_type, {})
    
    if version < parse_version("1.38.0"):
        return style_dict.get("v37", "")
    else:
        return style_dict.get("v39", "")


def get_combined_style(
        st_version:str, 
        *style_types:Literal["USER_CHAT", "ASSISTANT_CHAT", "RAG_USER_CHAT", "RAG_ASSISTANT_CHAT"]
    ) -> str:
    """
    获取多个样式类型的组合样式。
    
    :param st_version: Streamlit版本号, 形如 "1.37.0"
    :param style_types: 要组合的样式类型列表
    :return: 组合后的样式字符串
    """
    return "".join(get_style(style_type, st_version) for style_type in style_types)


i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

@lru_cache(maxsize=10)
def model_selector(model_type):
    if model_type == "OpenAI" or model_type == "AOAI":
        return ["gpt-3.5-turbo","gpt-3.5-turbo-16k","gpt-4","gpt-4-32k","gpt-4-1106-preview","gpt-4-vision-preview"]
    elif model_type == "Ollama":
        try:
           model_list = get_ollama_model_list() 
           return model_list
        except:
            return ["qwen:7b-chat"]
    elif model_type == "Groq":
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            model_list = get_groq_models(api_key=groq_api_key,only_id=True)

            # exclude tts model
            model_list_exclude_tts = [model for model in model_list if "whisper" not in model]
            excluded_models = [model for model in model_list if model not in model_list_exclude_tts]

            logger.info(f"Groq model list: {model_list}, excluded models:{excluded_models}")
            return model_list_exclude_tts
        except:
            logger.info("Failed to get Groq model list, using default model list")
            return ["llama3-8b-8192","llama3-70b-8192","llama2-70b-4096","mixtral-8x7b-32768","gemma-7b-it"]
    elif model_type == "Llamafile":
        return ["Noneed"]
    elif model_type == "LiteLLM":
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
# @st.cache_data
def write_chat_history(
    chat_history: Optional[List[Dict[str, str]]] = None,
    if_custom_css: bool = True
) -> None:
    # 将SVG编码为base64
    user_avatar = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"

    # 将SVG编码为base64
    ai_avatar = f"data:image/svg+xml;base64,{base64.b64encode(AI_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"

    if chat_history:
        for message in chat_history:
            try:
                if message["role"] == "system":
                    continue
            except:
                pass
            with st.chat_message(message["role"], avatar=user_avatar if message["role"] == "user" else ai_avatar):
                st.html(f"<span class='chat-{message['role']}'></span>")
                if isinstance(message["content"], str):
                    st.markdown(message["content"])
                elif isinstance(message["content"], List):
                    for content in message["content"]:
                        if content["type"] == "text":
                            st.markdown(content["text"])
                        elif content["type"] == "image_url":
                            # 如果开头为data:image/jpeg;base64，则解码为BytesIO对象
                            if content["image_url"]["url"].startswith("data:image/jpeg;base64"):
                                image_data = base64.b64decode(content["image_url"]["url"].split(",")[1])
                                st.image(image_data)
                            else:
                                st.image(content["image_url"])
        
        # 根据Streamlit版本选择样式
        if if_custom_css:
            chat_style = get_combined_style(st.__version__, "USER_CHAT", "ASSISTANT_CHAT")
            st.html(chat_style)

def wrap_long_text(text: str, max_length: int = 60) -> str:
    """
    将长文本按指定长度换行
    """
    wrapped_lines = textwrap.wrap(text, max_length)
    return '<br>'.join(wrapped_lines)

def export_chat_history_callback(
        chat_history: List[Dict[str, str]], 
        include_range: Optional[Tuple[int, int]] = None,
        exclude_indexes: Optional[List[int]] = None,
        is_rag: bool = False,
        export_type: Optional[str] = "html",
        theme: Optional[str] = "default",
        chat_name: Optional[str] = "Chat history",
        model_name: Optional[str] = None
    ):
    """
    导出聊天历史记录
    
    Args:
        chat_history (List[Dict[str, str]]): 聊天历史记录
        include_range (Optional[Tuple[int, int]]): 要包含的消息索引范围，例如 (0, 10)
        exclude_indexes (Optional[List[int]]): 要排除的消息索引列表，例如 [2, 5, 8]
        is_rag (bool, optional): 是否是RAG聊天记录. Defaults to False.
        export_type (str, optional): 导出类型，支持 "markdown" 和 "html". Defaults to "html".
        theme (str, optional): 导出主题. Defaults to "default".仅当export_type为html时有效
        chat_name (str, optional): 聊天记录的名称. Defaults to "Chat history".
        model_name (str, optional): 模型名称. Defaults to None.仅当export_type为html时有效
    """
    # 清理文件名，移除非法字符
    chat_name = re.sub(r'[\\/*?:"<>|]', "", chat_name).strip()
    if not chat_name:
        chat_name = "Chat history"

    export_folder = "chat histories export"
    os.makedirs(export_folder, exist_ok=True)

    if export_type == "markdown":
        markdown_content = generate_markdown_chat(
            chat_history=chat_history,
            include_range=include_range,
            exclude_indexes=exclude_indexes,
            chat_name=chat_name
        )

        filename = f"{'RAG ' if is_rag else ''}Chat history - {chat_name}.md"
        i = 1
        while os.path.exists(os.path.join(export_folder, filename)):
            filename = f"{'RAG ' if is_rag else ''}Chat history - {chat_name} ({i}).md"
            i += 1

        full_path = os.path.join(export_folder, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        st.toast(body=i18n(f"Chat history exported to: {full_path}"), icon="🎉")
    
    elif export_type == "html":
        html_content = generate_html_chat(
            chat_history=chat_history,
            include_range=include_range,
            exclude_indexes=exclude_indexes,
            theme=theme,
            chat_name=chat_name,
            model_name=model_name
        )

        filename = f"{'RAG ' if is_rag else ''}Chat history - {chat_name}.html"
        i = 1
        while os.path.exists(os.path.join(export_folder, filename)):
            filename = f"{'RAG ' if is_rag else ''}Chat history - {chat_name} ({i}).html"
            i += 1

        full_path = os.path.join(export_folder, filename)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        st.toast(body=i18n(f"Chat history exported to: {full_path}"), icon="🎉")

    # elif export_type == "jpg":
    #     image_stream = html_to_jpg(html_content)
    #     export_folder = "chat histories export"
    #     filename = "RAG Chat history.jpg" if is_rag else "Chat history.jpg"
    #     i = 1
    #     while os.path.exists(os.path.join(export_folder, filename)):
    #         filename = f"{'RAG ' if is_rag else ''}Chat history({i}).jpg"
    #         i += 1
        
    #     os.makedirs(export_folder, exist_ok=True)
    #     with open(os.path.join(export_folder, filename), "wb") as f:
    #         image_stream.save(f, format="JPEG")
    #     st.toast(body=i18n(f"Chat history exported to: " + os.path.join(export_folder, filename)), icon="🎉")
    else:
        st.error(i18n("Unsupported export type"))

def generate_markdown_chat(
    chat_history: List[Dict[str, str]], 
    include_range: Optional[Tuple[int, int]] = None, 
    exclude_indexes: Optional[List[int]] = None,
    chat_name: Optional[str] = "Chat history"
) -> str:
    """
    生成Markdown格式的聊天历史

    Args:
        chat_history (List[Dict[str, str]]): 完整的聊天历史
        include_range (Optional[Tuple[int, int]]): 要包含的消息索引范围，例如 (0, 10)
        exclude_indexes (Optional[List[int]]): 要排除的消息索引列表，例如 [2, 5, 8]

    Returns:
        str: 生成的Markdown格式聊天历史
    """
    formatted_history = []
    image_references = []
    image_counter = 0

    # 如果没有指定范围，则处理所有消息
    if include_range is None:
        include_range = (0, len(chat_history) - 1)
    
    # 如果没有指定排除索引，初始化为空列表
    if exclude_indexes is None:
        exclude_indexes = []

    formatted_history.append(f"# {chat_name}\n\n")

    for i in range(include_range[0], include_range[1] + 1):
        if i in exclude_indexes:
            continue

        message = chat_history[i]
        role = message['role'].title()
        content = message['content']
        
        formatted_history.append(f"## {role}\n\n")
        
        if isinstance(content, str):
            formatted_history.append(f"{content}\n\n")
        elif isinstance(content, list):
            for item in content:
                if item['type'] == 'text':
                    formatted_history.append(f"{item['text']}\n\n")
                elif item['type'] == 'image_url':
                    image_url = item['image_url'].get('url', '')
                    image_counter += 1
                    reference_id = f"image{image_counter}"
                    
                    if image_url.startswith('data:image/jpeg;base64,'):
                        formatted_history.append(f"![image][{reference_id}]\n\n")
                        image_references.append(f"[{reference_id}]: {image_url}\n")
                    else:
                        formatted_history.append(f"![Image]({image_url})\n\n")
    
    # 添加图片引用到文档末尾
    if image_references:
        formatted_history.append("\n\n<!-- Image References -->\n")
        formatted_history.extend(image_references)
    
    chat_history_text = "".join(formatted_history)

    return chat_history_text

def generate_html_chat(
        chat_history: List[Dict[str, str]], 
        include_range: Optional[Tuple[int, int]] = None, 
        exclude_indexes: Optional[List[int]] = None,
        theme: Optional[str] = "default",
        chat_name: Optional[str] = "Chat history",
        model_name: Optional[str] = None,
        code_theme: Optional[str] = "github-dark"
    ) -> str:
    """
    生成HTML格式的聊天历史，支持Markdown渲染，并使用指定的主题

    Args:
        chat_history (List[Dict[str, str]]): 完整的聊天历史
        include_range (Optional[Tuple[int, int]]): 要包含的消息索引范围，例如 (0, 10)
        exclude_indexes (Optional[List[int]]): 要排除的消息索引列表，例如 [2, 5, 8]
        theme (str, optional): 导出主题. Defaults to "default".
        chat_name (str, optional): 聊天记录的名称. Defaults to "Chat history".
        model_name (str, optional): 模型名称. Defaults to None.
        code_theme (str, optional): 代码主题. Defaults to "github-dark".可选值见 https://highlightjs.org/examples

    Returns:
        str: 生成的HTML格式聊天历史
    """
    if theme == "default":
        css_theme = default_theme
    elif theme == "glassmorphism":
        css_theme = glassmorphism_theme
    else:
        css_theme = default_theme  # 默认使用 default 主题

    html_template = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{chat_name}</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/{code_theme}.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        <style>
            {css_theme}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: #1A1B26;
                border-radius: 5px;
                padding: 1em;
                margin: 1em 0;
            }}
            code {{
                display: block;
                overflow-x: auto;
                padding: 0.5em;
                background-color: #1A1B26;
                font-family: 'Courier New', Courier, monospace;
                color: #CBD2EA;
            }}
            code[class*="language-"] {{
                background-color: #1A1B26;
                color: #CBD2EA;
            }}
            .message.user pre,
            .message.user code {{
                text-align: left;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="info-card">
                <div class="info-left">
                    <div class="project-name">RAGENT</div>
                    <div class="project-url">
                        <a href="https://github.com/Wannabeasmartguy/RAGENT" target="_blank">
                            github.com/Wannabeasmartguy/RAGENT
                        </a>
                    </div>
                </div>
                <div class="info-right">
                    <div class="info-right-content model-name">Model: {model_name}</div>
                    <div class="info-right-content message-count">Messages: {message_count}</div>
                    <div class="info-right-content chat-name">Chat: {chat_name}</div>
                </div>
            </div>
            <div class="chat-container">
                {chat_messages}
            </div>
        </div>
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {{
                marked.setOptions({{
                    breaks: true,
                    gfm: true,
                    highlight: function(code, lang) {{
                        const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                        return hljs.highlight(code, {{ language }}).value;
                    }},
                    langPrefix: 'hljs language-'
                }});
                
                document.querySelectorAll('.markdown-content').forEach((element) => {{
                    element.innerHTML = marked.parse(element.textContent);
                }});
                
                hljs.highlightAll();

                // 确保代码块内的换行符被保留
                document.querySelectorAll('pre code').forEach((block) => {{
                    block.innerHTML = block.innerHTML.replace(/\\n/g, '<br>');
                }});
            }});
        </script>
    </body>
    </html>
    """

    model_name = model_name if model_name is not None else "Not specified"
    wrapped_chat_name = wrap_long_text(chat_name)

    if include_range is None:
        include_range = (0, len(chat_history) - 1)
    if exclude_indexes is None:
        exclude_indexes = []
    
    # 计算消息数量，方法为：include_range范围长度 - exclude_indexes的元素个数
    message_count = include_range[1] - include_range[0] + 1 - len(exclude_indexes)

    chat_messages = []
    for i in range(include_range[0], include_range[1] + 1):
        if i in exclude_indexes:
            continue

        message = chat_history[i]
        role = message['role']
        content = message['content']
        
        message_html = f'<div class="message {role}">'
        message_html += f'<div class="role">{role.capitalize()}</div>'
        
        if isinstance(content, str):
            message_html += f'<div class="markdown-content">{content}</div>'
        elif isinstance(content, list):
            for item in content:
                if item['type'] == 'text':
                    message_html += f'<div class="markdown-content">{item["text"]}</div>'
                elif item['type'] == 'image_url':
                    image_url = item['image_url'].get('url', '')
                    message_html += f'<img src="{image_url}" alt="Image">'
        
        message_html += '</div>'
        chat_messages.append(message_html)
    
    # 使用 str.replace() 来插入 CSS 主题
    html_content = html_template.replace("{css_theme}", css_theme)
    # 然后使用 .format() 插入聊天消息
    html_content = html_content.format(
        chat_messages="\n".join(chat_messages),
        chat_name=wrapped_chat_name,
        model_name=model_name,
        message_count=message_count,
        code_theme=code_theme
    )
    
    return html_content

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


def copy_to_clipboard(content: str):
    '''
    将内容复制到剪贴板,并提供streamlit提醒
    '''
    pyperclip.copy(content)
    st.toast(i18n("The content has been copied to the clipboard"), icon="✂️")


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
    构造用户输入的字典。
    """
    base_input = {
        "role": "user"
    }

    if images is None:
        base_input["content"] = prompt
    elif isinstance(images, (BytesIO, list)):
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