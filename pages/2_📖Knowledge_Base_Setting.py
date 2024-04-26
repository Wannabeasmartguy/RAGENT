import streamlit as st
import streamlit.components.v1 as components
import os

from configs.basic_config import I18nAuto
from configs.knowledge_base_config import KnowledgeBase, SubKnowledgeBase
from utils.chroma_utils import *


i18n = I18nAuto()
kbs = KnowledgeBase()

with st.sidebar:
    # 获得同级文件夹 /img 的路径
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
    st.image(logo_path)

    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.write("---")


file_upload = st.file_uploader(label=i18n("Upload File"),
                               accept_multiple_files=True,
                               label_visibility="collapsed")
file_uploaded_name_list = [file.name for file in file_upload]

st.write(i18n("### Choose the knowledge base you want to use"))

kb_choose = st.selectbox(label=i18n("Knowledge Base Choose"), 
                            #  label_visibility="collapsed",
                             options=kbs.knowledge_bases,
                             key="kb_choose")
kb = SubKnowledgeBase(kb_choose)

column1, column2 = st.columns([0.7,0.3])
with column1:
    file_names_inchroma = st.selectbox(label=i18n("Files in Knowledge Base"),
                                       label_visibility="collapsed",
                                       options=load_vectorstore(persist_vec_path=kb.persist_vec_path,
                                                                embedding_model_type=kb.embedding_model_type,
                                                                embedding_model=kb.embedding_model))
with column2:
    get_knowledge_base_info_button = st.button(label=i18n("Get Knowledge Base info"))


if get_knowledge_base_info_button:
    chroma_info_html = get_chroma_file_info(persist_path=kb.persist_vec_path,
                    file_name=file_names_inchroma,
                    advance_info=False)
    components.html(chroma_info_html,
                    height=800)

# st.write(vars(kb))
# st.html(
#     '''
#     <div class="title_txt">
#         <div class="title">Source</div>
#     </div>
#     <!DOCTYPE html>
#     <html lang="zh-CN">
#     <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>搜索结果卡片</title>
#     <style>
#         .card-container {
#             display: flex;
#             flex-wrap: wrap;
#         }
#         .card {
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             padding: 10px;
#             margin: 10px;
#             background-color: #f9f9f9;
#             flex: 1 0 25%;
#             transition: box-shadow 0.3s;
#             width: 150px; /* 或者使用百分比宽度，如 width: 100%; */
#             box-sizing: border-box; /* 确保宽度包括边框和内边距 */
#             text-decoration: none; /* 移除链接的下划线 */
#             color: inherit;
#         }
#         .card:hover {
#             box-shadow: 0 0 11px rgba(33,33,33,.2); 
#         }
#         .card img {
#             width: 80px;
#             height: 80px;
#             object-fit: cover;
#             border-radius: 4px;
#         }
#         .card .title {
#             font-weight: bold;
#             margin-bottom: 5px;
#         }
#         .card .snippet {
#             color: #666;
#             font-size: 14px;
#             overflow: hidden;
#             text-overflow: ellipsis;
#             white-space: nowrap;
#         }
#         .card .url {
#             font-size: 12px;
#             color: #999;
#             text-overflow: ellipsis;
#             overflow: hidden;
#         }
#         .title_txt {
#             font-size: 18px;
#             font-weight: bold;
#             color: #3b82f6;
#         }
#     </style>
#     </head>
#     <body>
#     <div class="card-container">
#     <a href="http://localhost:8501/" class="card">
        
#         <div class="card-content">
#             <div class="title">test</div>
#             <div class="url">"http://localhost:8501/"</div>
#         </div>
#     </a>
#     </div></body></html>
#     '''
# )