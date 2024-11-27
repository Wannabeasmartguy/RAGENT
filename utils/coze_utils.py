import json
from datetime import datetime
from typing import Dict

from core.processors.chat.classic import CozeChatProcessor
from core.model.coze.bot import Bot
from core.basic_config import I18nAuto, SUPPORTED_LANGUAGES
import streamlit as st

i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

def json_to_botcard(json_data: str | Dict):
    """
    Convert a JSON string or dictionary to a styled HTML card representation with plugins.

    Parameters:
    - json_data (str/dict): A JSON string or dictionary containing bot details.

    Returns:
    - str: An HTML string representing the bot information as a styled card with plugins.
    """

    # 如果输入是字符串，尝试将其解析为字典
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return "Invalid JSON string."
    else:
        data = json_data

    # 获取data字段中的信息
    bot_info = data.get('data', {})

    # 将时间戳转换为可读的日期格式
    def format_timestamp(ts):
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else 'N/A'

    # 生成插件信息的HTML
    plugin_html = ""
    for plugin in bot_info.get('plugin_info_list', []):
        plugin_html += f"""
            <div class="plugin">
                <img src="{plugin.get('icon_url', '')}" alt="{plugin.get('name', 'No Plugin Name')}" class="plugin-icon">
                <div class="plugin-details">
                    <h3>{plugin.get('name', 'No Plugin Name')}</h3>
                    <p>{plugin.get('description', 'No Description')}</p>
                </div>
            </div>
        """

    # 创建带样式的HTML卡片
    html_card = f"""
    <html>
    <head>
        <style>
            .bot-card {{
                border: 1px solid #ddd;
                border-radius: 15px;
                display: flex;
                align-items: center;
                padding: 20px;
                margin: 20px;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
                flex-wrap: wrap; /* 使内容在必要时换行 */
            }}
            .bot-details, .plugin-details {{
                flex: 1;
            }}
            .bot-icon, .plugin-icon {{
                width: 100px;
                height: 100px;
                border-radius: 50%;
                margin-left: 20px;
                margin-right: 20px;
            }}
            .plugin {{
                display: flex;
                border: 1px solid #ddd;
                border-radius: 15px;
                align-items: center;
                margin-top: 10px; /* 插件之间的间隔 */
            }}
            .plugins-container {{
                width: 100%;
                border-top: 1px solid #ddd;
                padding-top: 10px;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="bot-card">
            <div class="bot-details">
                <h2>{bot_info.get('name', 'No Name')}</h2>
                <p>{bot_info.get('description', 'No Description')}</p>
                <p><strong>Created:</strong> {format_timestamp(bot_info.get('create_time'))}</p>
                <p><strong>Updated:</strong> {format_timestamp(bot_info.get('update_time'))}</p>
                <p><strong>Version:</strong> {bot_info.get('version', 'N/A')}</p>
                <p><strong>Prompt:</strong> {bot_info.get('prompt_info', {}).get('prompt', 'N/A')}</p>
                <!-- 其他详细信息可以继续添加在这里 -->
            </div>
            <img src="{bot_info.get('icon_url', '')}" alt="Bot Icon" class="bot-icon">
            <div class="plugins-container">
                {plugin_html}
            </div>
        </div>
    </body>
    </html>
    """

    return html_card


def display_bot_info(
        access_token: str,
        bot_id: str
    ):
    if not access_token or not bot_id:
        st.error(i18n("Please provide the access token and bot ID in sidebar."))
        return

    bot_info = CozeChatProcessor.get_bot_config(access_token,bot_id)
    bot_model = Bot(**bot_info.json())
    bot = bot_model.model_dump_json()

    bot_card = json_to_botcard(bot)
    st.html(bot_card)