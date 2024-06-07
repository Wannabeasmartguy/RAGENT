import json
from datetime import datetime
from typing import Dict

def json_to_botcard(json_data: str | Dict):
    """
    Convert a JSON string or dictionary to a styled HTML card representation.

    Parameters:
    - json_data (str/dict): A JSON string or dictionary containing bot details.

    Returns:
    - str: An HTML string representing the bot information as a styled card.
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
                width: 550px;
            }}
            .bot-details {{
                flex-grow: 1;
            }}
            .bot-icon {{
                width: 100px;
                height: 100px;
                border-radius: 50%;
                margin-left: 20px;
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
                <!-- 其他详细信息可以继续添加在这里 -->
            </div>
            <img src="{bot_info.get('icon_url', '')}" alt="Bot Icon" class="bot-icon">
        </div>
    </body>
    </html>
    """

    return html_card