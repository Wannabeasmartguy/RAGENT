
import os
import re
import time
import base64
from typing import List, Dict, Optional, Literal, Tuple

from config.constants import (
    I18N_DIR, 
    SUPPORTED_LANGUAGES,
    USER_AVATAR_SVG,
    AI_AVATAR_SVG
)
from assets.styles.css.rag_chat_css import (
    RAG_CHAT_USER_STYLE_ST_V37,
    RAG_CHAT_USER_STYLE_ST_V39,
    RAG_CHAT_ASSISTANT_STYLE_ST_V37,
    RAG_CHAT_ASSISTANT_STYLE_ST_V39,
)
from assets.styles.css.classic_chat_css import (
    USER_CHAT_STYLE_ST_V37,
    USER_CHAT_STYLE_ST_V39,
    ASSISTANT_CHAT_STYLE,
)
from assets.styles.css.export_themes import (
    default_theme, 
    glassmorphism_theme
)
from core.basic_config import I18nAuto
from tools.toolkits import TO_TOOLS

import streamlit as st
import streamlit.components.v1 as components
import whisper
import pyperclip
import textwrap
from streamlit_float import *
from pkg_resources import parse_version


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES[language]
)

SCROLL_BUTTON_CONSTANTS = {
    "BACK_TO_TOP": {
        "v37": '''
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            body.scrollTop = 0;
        </script>
        ''',
        "v39": '''
        <script>
            var body = window.parent.document.querySelector(".stMain");
            console.log(body);
            body.scrollTop = 0;
        </script>
        '''
    },
    "BACK_TO_BOTTOM": {
        "v37": '''
        <script>
            var body = window.parent.document.querySelector(".main");
            console.log(body);
            body.scrollTop = body.scrollHeight;
        </script>
        ''',
        "v39": '''
        <script>
            var body = window.parent.document.querySelector(".stMain");
            console.log(body);
            body.scrollTop = body.scrollHeight;
        </script>
        '''
    }
}

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


def get_scroll_button_js(
    scroll_type: Literal["BACK_TO_TOP", "BACK_TO_BOTTOM"],
    st_version: str
):
    """
    Get the scroll button js code.
    
    Args:
        version (str): The version of the streamlit.Like "1.37" or "1.39".
    
    Returns:
        str: The scroll button js code.
    """
    version = parse_version(st_version)
    if version < parse_version("1.38.0"):
        return SCROLL_BUTTON_CONSTANTS[scroll_type]["v37"]
    else:
        return SCROLL_BUTTON_CONSTANTS[scroll_type]["v39"]

def get_scroll_button_collection(
    st_version: str
):
    """
    Get the both scroll buttons js code.
    
    Args:
        version (str): The version of the streamlit.Like "1.37" or "1.39".
    
    Returns:
        dict: A dict containing the scroll button js code.
    """
    return dict(
        back_to_top=get_scroll_button_js(scroll_type="BACK_TO_TOP", st_version=st_version),
        back_to_bottom=get_scroll_button_js(scroll_type="BACK_TO_BOTTOM", st_version=st_version)
    )

def copy_to_clipboard(content: str):
    '''
    将内容复制到剪贴板,并提供streamlit提醒
    '''
    pyperclip.copy(content)
    st.toast(i18n("The content has been copied to the clipboard"), icon="✂️")

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
                if "reasoning_content" in message and message["reasoning_content"]:
                    st.caption(message['reasoning_content'])
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

    if chat_name:
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

def back_to_top(script_container = st.empty(), buttom_container = st.empty()):
    """
    Scroll the page to the top.
    
    Args:
        script_container (streamlit.empty, optional): The temporary container to hold the script. Defaults to st.empty().
        buttom_container (streamlit.empty, optional): The container to hold the button. Defaults to st.empty().
    """
    top_container = buttom_container.container()
    top_css = float_css_helper(width="2.2rem", right="10rem", bottom="13rem")
    with top_container:
        up_button = st.button("⭱", key="up_button")
        if up_button:
            with script_container:
                components.html(get_scroll_button_collection(st_version=st.__version__)["back_to_top"])
                time.sleep(.5) # To make sure the script can execute before being deleted
            script_container.empty()
    top_container.float(top_css)


def back_to_bottom(script_container = st.empty(), buttom_container = st.empty()):
    """
    Scroll the page to the bottom.
    
    Args:
        script_container (streamlit.empty, optional): The temporary container to hold the script. Defaults to st.empty().
        buttom_container (streamlit.empty, optional): The container to hold the button. Defaults to st.empty().
    """
    bottom_container = buttom_container.container()
    bottom_css = float_css_helper(width="2.2rem", right="10rem", bottom="10rem")
    with bottom_container:
        bottom_button = st.button("⭳", key="bottom_button")
        if bottom_button:
            with script_container:
                components.html(get_scroll_button_collection(st_version=st.__version__)["back_to_bottom"])
                time.sleep(.5) # To make sure the script can execute before being deleted
            script_container.empty()
    bottom_container.float(bottom_css)


def float_chat_input_with_audio_recorder(if_tools_call: str = False, prompt_disabled: bool = False) -> str:
    """
    Create a container with a floating chat input and an audio recorder.

    Returns:
        str: The text input from the user.
    """        
    # Create a container with a floating chat input and an audio recorder
    chat_input_container = st.container()
    with chat_input_container:
        # divide_context_column, character_input_column, voice_input_column = st.columns([0.1,0.9,0.1])
        if if_tools_call:
            tools_popover = st.popover(label="🔧")
            tools_popover.multiselect(
                label=i18n("Functions"),
                options=TO_TOOLS.keys(),
                default=list(TO_TOOLS.keys())[:2],
                help=i18n("Select functions you want to use."),
                # format_func 将所有名称开头的"tool_"去除
                format_func=lambda x: x.replace("tool_","").replace("_"," "),
                key="tools_popover"
            )
        character_input_column, voice_input_column = st.columns([0.9,0.1])
        # divide_context_placeholder = divide_context_column.empty()
        # divide_context_button = divide_context_placeholder.button(
        #     label="✂️",
        # )
        # if divide_context_button:
        #     storage.upsert()

        # the chat input in the middle
        st.markdown(
            """
            <style> 
                .stChatInput > div {
                    background-color: #FFFFFF;
                    border-radius: 10px;
                    border: 1px solid #E0E0E0;
                }
            </style>
            """, 
            unsafe_allow_html=True
        )
        character_input_placeholder = character_input_column.empty()
        prompt = character_input_placeholder.chat_input("What is up?", disabled=prompt_disabled)

        # the button (actually popover) on the right side of the chat input is to record audio
        voice_input_popover = voice_input_column.popover(
            label="🎤"
        )
        transcribe_model_name = voice_input_popover.selectbox(
            label=i18n("Transcribe model"),
            options=whisper.available_models(),
            index=3,
            key="transcribe_model"   
        )
        audio_recorder_container =  voice_input_popover.container(border=True)
        with audio_recorder_container:
            audio_recorded = st.audio_input(label=i18n("Record your input"))
            transcribe_button_placeholder = st.empty()
            if audio_recorded:
                transcribe_button = transcribe_button_placeholder.button(
                    label=i18n("Transcribe"),
                    use_container_width=True
                )
                # 临时存储音频文件,将BytesIO对象转换为文件对象
                with open("dynamic_configs/temp.wav", "wb") as f:
                    f.write(audio_recorded.getvalue())
                # 加载语音识别模型
                if transcribe_button:
                    with st.status(i18n("Transcribing...")):
                        st.write(i18n("Loading model"))
                        transcribe_model = whisper.load_model(
                            name=transcribe_model_name,
                            download_root="./tts_models"
                        )
                        st.write(i18n("Model loaded"))
                        # 识别语音
                        st.write(i18n("Transcribing"))
                        transcribe_result = transcribe_model.transcribe(audio="dynamic_configs/temp.wav",word_timestamps=True,verbose=True)
                        st.write(i18n("Transcribed"))
                    content = transcribe_result.get("text","No result.")
                    copy_to_clipboard(content)
                    st.code(content)
                    # 删除临时文件
                    os.remove("dynamic_configs/temp.wav")

    chat_input_css = float_css_helper(bottom="5rem", display="flex", justify_content="center", margin="0 auto")
    chat_input_container.float(chat_input_css)
    return prompt


@st.fragment
def define_fragment_image_uploader(
    key: str,
):
    return st.file_uploader(
        label=i18n("Upload images"),
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=key,
    )

@st.dialog(title=i18n("Export Settings"), width="large")
def export_dialog(
    chat_history: List[Dict],
    is_rag: bool = False,
    chat_name: str = "Chat history",
    model_name: Optional[str] = None
):
    with st.container(border=True):
        export_type = st.selectbox(
            label=i18n("Export File Type"),
            options=["markdown", "html"],
            format_func=lambda x: x.title()
        )
        if export_type == "html":
            export_theme = st.selectbox(
                label=i18n("Please select the theme to use"),
                options=["default", "glassmorphism"],
                format_func=lambda x: x.title()
            )

        if len(chat_history) > 2:
            with st.expander(i18n("Advanced Options")):
                include_range = st.select_slider(
                    label=i18n("Include range"),
                    options=["All", "Custom"],
                    value="All"
                )
                if include_range == "Custom":
                    start_position_column, end_position_column = st.columns(2)
                    with start_position_column:
                        # 起始位置，从1开始（符合用户习惯）
                        start_position = st.slider(
                            label=i18n("Start Position"),
                            min_value=1,
                            max_value=len(chat_history)-1,
                            value=1,
                        )
                    with end_position_column:
                        # 结束位置，从1开始（符合用户习惯）
                        end_position = st.slider(
                            label=i18n("End Position"),
                            min_value=start_position,
                            max_value=len(chat_history),
                            value=len(chat_history)
                        )

                    # 动态生成 exclude_indexes 的选项
                    exclude_options = list(range(start_position, end_position + 1))
                    
                    # 排除的对话消息索引
                    exclude_indexes = st.multiselect(
                        label=i18n("Exclude indexes"),
                        options=exclude_options,
                        default=[],
                        format_func=lambda x: f"Message {x}",
                        placeholder=i18n("Selected messages will not be exported")
                    )
        else:
            include_range = "All"

        export_submit_button = st.button(
            label=i18n("Export"),
            use_container_width=True,
            type="primary"
        )
    
    if export_type == "markdown":
        from utils.st_utils import generate_markdown_chat
        # 传入的值从1开始，但要求传入的值从0开始
        preview_content = generate_markdown_chat(
            chat_history=chat_history,
            chat_name=chat_name,
            include_range=(start_position-1, end_position-1) if include_range == "Custom" else None,
            exclude_indexes=[x-1 for x in exclude_indexes] if include_range == "Custom" else None,
        )
        with st.expander(i18n("Preview")):
            content_preview = st.markdown(preview_content)
    elif export_type == "html":
        from utils.st_utils import generate_html_chat
        # 传入的值从1开始，但要求传入的值从0开始
        preview_content = generate_html_chat(
            chat_history=chat_history,
            chat_name=chat_name,
            model_name=model_name,
            include_range=(start_position-1, end_position-1) if include_range == "Custom" else None,
            exclude_indexes=[x-1 for x in exclude_indexes] if include_range == "Custom" else None,
            theme=export_theme
        )
        with st.expander(i18n("Preview")):
            st.info(i18n("Background appearance cannot be previewed in real time due to streamlit limitations, please click the submit button to export and check the result."))
            content_preview = st.html(preview_content)
    # elif export_type == "jpg":
    #     from utils.basic_utils import html_to_jpg
    #     preview_content = html_to_jpg(chat_history)
    #     with st.expander(i18n("Preview")):
    #         st.info(i18n("Background appearance cannot be previewed in real time due to streamlit limitations, please click the submit button to export and check the result."))
    #         content_preview = st.image(preview_content)

    if export_submit_button:
        # 传入的值从1开始，但要求传入的值从0开始
        export_chat_history_callback(
            chat_history=chat_history,
            include_range=(start_position-1, end_position-1) if include_range == "Custom" else None,
            exclude_indexes=[x-1 for x in exclude_indexes] if include_range == "Custom" else None,
            is_rag=is_rag,
            export_type=export_type,
            theme=export_theme if export_type == "html" else None,
            chat_name=chat_name,
            model_name=model_name
        )