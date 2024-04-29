import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
import os
import tempfile

from configs.basic_config import I18nAuto
from configs.knowledge_base_config import KnowledgeBase, SubKnowledgeBase, create_vectorstore
from utils.chroma_utils import *
from utils.text_splitter.text_splitter_utils import *


i18n = I18nAuto()
kbs = KnowledgeBase()
openai_embedding_model = ["text-embedding-ada-002"]
local_embedding_model = ['bge-base-zh-v1.5','bge-base-en-v1.5',
                         'bge-large-zh-v1.5','bge-large-en-v1.5']

def embed_model_selector(embed_model_type):
    if embed_model_type == "OpenAI":
        return ["text-embedding-ada-002"]
    elif embed_model_type == "Hugging Face(local)":
        return ['bge-base-zh-v1.5','bge-base-en-v1.5',
                'bge-large-zh-v1.5','bge-large-en-v1.5']
    else:
        return None

if "embed_model_type" not in st.session_state:
    st.session_state.embed_model_type = "Hugging Face(local)"

if "pages" not in st.session_state:
    st.session_state.pages = []

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

with st.sidebar:
    # 获得同级文件夹 /img 的路径
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
    st.image(logo_path)

    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.write("---")

    embed_model_type_selectbox = st.selectbox(label=i18n("Embed Model Type"),
                                              options=["OpenAI", "Hugging Face(local)"],
                                              key="embed_model_type")
    embed_model_selectbox = st.selectbox(label=i18n("Embed Model"),
                                         options=embed_model_selector(st.session_state.embed_model_type),
                                         key="embed_model")
    # create_kb_button = st.button(label=i18n("Create Knowledge Base"),
    #                                on_click=create_vectorstore(embed_model_selectbox))
    

kb_choose = st.selectbox(label=i18n("Knowledge Base Choose"), 
                         #  label_visibility="collapsed",
                         options=kbs.knowledge_bases,
                         key="kb_choose")
kb = SubKnowledgeBase(kb_choose)

reinitialize_kb_button = st.button(label=i18n("Reinitialize Knowledge Base"),
                                   on_click=kb.reinitialize(kb_choose))


st.write("## Choose files you want to embed")

file_upload = st.file_uploader(label=i18n("Upload File"),
                               accept_multiple_files=True,
                               label_visibility="collapsed",
                               key=st.session_state["file_uploader_key"],)

with st.expander(label=i18n("File Handling Configuration"), expanded=False):
    chunk_size_column, overlap_column = st.columns(2)
    with chunk_size_column:
        split_chunk_size = st.number_input(label=i18n("Chunk Size"),
                                           value=1000,
                                           step=1)
    with overlap_column:
        split_overlap = st.number_input(label=i18n("Overlap"),
                                        value=0,
                                        step=1)

upload_column,clear_column = st.columns([0.7,0.3])

with upload_column:
    upload_and_split_file_button = st.button(label=i18n("① Upload and Split Files"),
                                         use_container_width=True)

with clear_column:
    clear_file_button = st.button(label=i18n("Clear File"),use_container_width=True)
    if clear_file_button:
        st.session_state["file_uploader_key"] += 1
        st.session_state.pages = []
        st.experimental_rerun()

embed_button = st.button(label=i18n("② Embed Files"),use_container_width=True,type="primary")
        
preview_placeholder = st.empty()

if file_upload:
    if upload_and_split_file_button:
        pages = []
        for file in file_upload:
            file_name = file.name
            # 获取文件类型，以在创建临时文件时使用正确的后缀
            file_suffix = Path(file.name).suffix
            # 获取文件名，不包含后缀
            file_name_without_suffix = Path(file.name).stem

            # delete 设置为 False,才能在解除绑定后使用 temp_file 进行分割
            with tempfile.NamedTemporaryFile(prefix=file_name_without_suffix + "__",suffix=file_suffix,delete=False) as temp_file:
                # stringio = StringIO(file.getvalue().decode())
                temp_file.write(file.getvalue())
                temp_file.seek(0)
                # st.write(temp_file.name)
                # st.write("File contents:")
                # st.write(temp_file.read())
            
            splitted_docs = choose_text_splitter(file_path=temp_file,chunk_size=split_chunk_size,chunk_overlap=split_overlap)
            # 手动删除临时文件
            os.remove(temp_file.name)
            # st.write(splitted_docs[0].page_content)
            # st.write(splitted_docs)

            pages.extend(splitted_docs)
            st.session_state.pages = pages

    option = preview_placeholder.selectbox(
        label=i18n("Choose the file you want to preview"),
        # label_visibility="collapsed",
        options=st.session_state.pages,
    )
    with st.container(border=True):
        if option:
            st.write(option.page_content)
else:
    file_upload.clear()
    preview_placeholder.empty()
    st.session_state.pages = []


if embed_button:
    if len(st.session_state.pages) == 0:
        st.warning(i18n("Please upload and split files first"))
    else:
        with st.spinner():
            kb.add_file_in_vectorstore(split_docs=st.session_state.pages, file_obj=file_upload)
        st.toast("Embedding completed!")


st.write("---")
st.write(f"知识库 `{kb_choose}` 中的文件：")

column1, column2 = st.columns([0.7,0.3])
with column1:
    file_names_inchroma = st.selectbox(label=i18n("Files in Knowledge Base"),
                                       label_visibility="collapsed",
                                       options=kb.load_file_names)
with column2:
    get_knowledge_base_info_button = st.button(label=i18n("Get Knowledge Base info"))

delete_file_button = st.button(label=i18n("Delete the File"),
                               use_container_width=True)

if get_knowledge_base_info_button:
    chroma_info_html = get_chroma_file_info(persist_path=kb.persist_vec_path,
                    file_name=file_names_inchroma,
                    advance_info=False)
    components.html(chroma_info_html,
                    height=800)

if delete_file_button:
    kb.delete_flie_in_vectorstore(files_samename=file_names_inchroma)
    st.rerun()