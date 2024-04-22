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
st.write("contain setting for agent chat")