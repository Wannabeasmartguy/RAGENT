# CSS 主题集合

default_theme = """
body {{
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f0f0f0;
}}
.chat-title {{
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    color: #333;
    margin-top: 30px;
}}
.chat-subtitle {{
    font-size: 16px;
    text-align: center;
    margin-bottom: 20px;
    color: #666;
    margin-bottom: 30px;
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
.chat-title {{
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
    color: #333;
    margin-top: 20px;
}}
.chat-subtitle {{
    font-size: 16px;
    text-align: center;
    margin-bottom: 20px;
    color: #666;
    margin-bottom: 30px;
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