# CSS 主题集合

default_theme = """
body {{
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f0f0f0;
}}
.card {{
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
}}
.info-card {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    padding: 15px;
    margin-bottom: 20px;
}}
.info-left {{
    text-align: left;
}}
.info-right {{
    text-align: right;
}}
.info-right-content {{
    background-color: #ffffff;
    border-radius: 8px;
    padding: 4px 12px;
    margin-bottom: 3px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    text-align: left;
    max-width: 300px;
    word-wrap: break-word;
}}
.project-name {{
    font-size: 20px;
    font-weight: bold;
    color: #333;
}}
.project-url {{
    font-size: 14px;
    color: #666;
}}
.project-url a {{
    color: #007bff;
    text-decoration: none;
}}
.model-name, .message-count, .chat-name {{
    font-size: 14px;
    color: #666;
    margin-bottom: 5px;
}}
.message {{
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 10px;
}}
.user {{
    background-color: #E7F8FF;
    text-align: right;
    padding-right: 20px;
}}
.assistant {{
    background-color: #F7F8FA;
    padding-left: 20px;
}}
.role {{
    font-weight: bold;
    margin-bottom: 5px;
}}
img {{
    max-width: 100%;
    height: auto;
}}
pre {{
    background-color: #f4f4f4;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}}
code {{
    font-family: 'Courier New', Courier, monospace;
}}
"""

glassmorphism_theme = """
body {{
    background-color: #50E3C2;
    background-image: 
        radial-gradient(at 47% 33%, hsl(162.00, 77%, 40%) 0, transparent 59%), 
        radial-gradient(at 82% 65%, hsl(198.00, 100%, 50%) 0, transparent 55%);
    font-family: Arial, sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 20px;
}}
.card {{
    background-color: #ffffff;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    padding: 20px;
}}
.info-card {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: rgba(248, 249, 250, 0.8);
    border: 2px solid rgba(0, 0, 0, 0.1);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
}}
.info-left {{
    text-align: left;
}}
.info-right {{
    text-align: right;
}}
.info-right-content {{
    background-color: #ffffff;
    border-radius: 8px;
    padding: 4px 12px;
    margin-bottom: 3px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    text-align: left;
    max-width: 300px;
    word-wrap: break-word;
}}
.project-name {{
    font-size: 20px;
    font-weight: bold;
    color: #333;
}}
.project-url {{
    font-size: 14px;
    color: #666;
}}
.project-url a {{
    color: #007bff;
    text-decoration: none;
}}
.model-name, .message-count, .chat-name {{
    font-size: 14px;
    color: #666;
    margin-bottom: 5px;
}}
.card {{
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    background-color: rgba(255, 255, 255, 0.75);
    border-radius: 12px;
    border: 1px solid rgba(209, 213, 219, 0.3);
    padding: 20px;
    max-width: 800px;
    width: 100%;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}}
.message {{
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 10px;
}}
.user {{
    background-color: rgba(231, 248, 255, 0.6);
    text-align: right;
    padding-right: 20px;
}}
.assistant {{
    background-color: rgba(247, 248, 250, 0.6);
    padding-left: 20px;
}}
.role {{
    font-weight: bold;
    margin-bottom: 5px;
}}
img {{
    max-width: 100%;
    height: auto;
}}
pre {{
    background-color: rgba(244, 244, 244, 0.6);
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
}}
code {{
    font-family: 'Courier New', Courier, monospace;
}}
"""

# 可以继续添加更多主题...