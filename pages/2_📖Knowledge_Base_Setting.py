import streamlit as st
import os


with st.sidebar:
    # 获得同级文件夹 /img 的路径
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
    st.image(logo_path)

    st.page_link("RAGenT.py", label="💭 Chat")
    st.page_link("pages/1_🤖AgentChat_Setting.py", label="🤖 AgentChat Setting")
    st.page_link("pages/2_📖Knowledge_Base_Setting.py", label="📖 Knowledge_Base_Setting")

st.write("placeholder, will implement later")
st.write("Porting GGA's knowledge base management features")

st.html(
    '''
    <div class="title_txt">
        <div class="title">Source</div>
    </div>
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>搜索结果卡片</title>
    <style>
        .card-container {
            display: flex;
            flex-wrap: wrap;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin: 10px;
            background-color: #f9f9f9;
            flex: 1 0 25%;
            transition: box-shadow 0.3s;
            width: 150px; /* 或者使用百分比宽度，如 width: 100%; */
            box-sizing: border-box; /* 确保宽度包括边框和内边距 */
            text-decoration: none; /* 移除链接的下划线 */
            color: inherit;
        }
        .card:hover {
            box-shadow: 0 0 11px rgba(33,33,33,.2); 
        }
        .card img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
        }
        .card .title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .card .snippet {
            color: #666;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .card .url {
            font-size: 12px;
            color: #999;
            text-overflow: ellipsis;
            overflow: hidden;
        }
        .title_txt {
            font-size: 18px;
            font-weight: bold;
            color: #3b82f6;
        }
    </style>
    </head>
    <body>
    <div class="card-container">
    <a href="http://localhost:8501/" class="card">
        
        <div class="card-content">
            <div class="title">test</div>
            <div class="url">"http://localhost:8501/"</div>
        </div>
    </a>
    </div></body></html>
    '''
)