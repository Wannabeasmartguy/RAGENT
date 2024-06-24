import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
import os
import tempfile
import json

from configs.basic_config import I18nAuto,SUPPORTED_LANGUAGES
from configs.knowledge_base_config import ChromaCollectionProcessor, ChromaVectorStoreProcessor
from utils.chroma_utils import *
from utils.text_splitter.text_splitter_utils import *

from api.routers.knowledgebase import KNOWLEDGE_BASE_PATH


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])

embedding_dir = "embeddings"

openai_embedding_model = ["text-embedding-ada-002"]
local_embedding_model = ['bge-base-zh-v1.5','bge-base-en-v1.5',
                         'bge-large-zh-v1.5','bge-large-en-v1.5']

def embed_model_selector(embed_model_type):
    if embed_model_type == "openai":
        return openai_embedding_model
    elif embed_model_type == "huggingface":
        return local_embedding_model
    else:
        return None

if "embed_model_type" not in st.session_state:
    st.session_state.embed_model_type = "huggingface"

if "pages" not in st.session_state:
    st.session_state.pages = []

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# set a counter to update the st.cache_data of collections in Chroma
if "collection_counter" not in st.session_state:
    st.session_state.collection_counter = 0

# set a counter to update the st.cache_data of documents in Chroma collection
if "document_counter" not in st.session_state:
    st.session_state.document_counter = 0

with st.sidebar:
    # 获得同级文件夹 /img 的路径
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
    st.image(logo_path)

    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.write("---")
    st.info(i18n("Please choose embedding model when creating the knowledge base"))

    embed_model_type_selectbox = st.selectbox(
        label=i18n("Embed Model Type"),
        options=["openai", "huggingface"],
        key="embed_model_type"
    )
    embed_model_selectbox = st.selectbox(
        label=i18n("Embed Model"),
        options=embed_model_selector(st.session_state.embed_model_type),
        key="embed_model"
    )
    
    with st.popover(label=i18n("Local embedding model download"),use_container_width=True):
        huggingface_repo_id_input = st.text_input(
            label=i18n("Huggingface repo id"),
            placeholder=i18n("Paste huggingface repo id here"),
        )

        embed_model_download_button = st.button(
            label=i18n("Download embedding model"),
        )
    # create_kb_button = st.button(label=i18n("Create Knowledge Base"),
    #                                on_click=create_vectorstore(embed_model_selectbox))


# 根据嵌入模型的选择情况，创建ChromaVectorStoreProcessor
if embed_model_type_selectbox == "openai":
    chroma_vectorstore_processor = ChromaVectorStoreProcessor(
        embedding_model_name_or_path=embed_model_selectbox,
        embedding_model_type=embed_model_type_selectbox,
        api_key=os.getenv("AZURE_OAI_KEY",default="OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OAI_ENDPOINT",default=None),
        api_version=os.getenv("API_VERSION",default="2024-02-15-preview"),
        api_type=os.getenv("API_TYPE",default="openai"),
    )
elif embed_model_type_selectbox == "huggingface":
    chroma_vectorstore_processor = ChromaVectorStoreProcessor(
        embedding_model_name_or_path=os.path.join(embedding_dir,embed_model_selectbox),
        embedding_model_type=embed_model_type_selectbox,
    )
    # 检查是否已有本地嵌入模型
    embed_model_situation = chroma_vectorstore_processor.model_dir_verify(
        model_name_or_path=os.path.join(embedding_dir,embed_model_selectbox)
    )
    if embed_model_situation:
        st.toast(embed_model_situation, icon="⚠️")


# 创建了chroma_vectorstore_handler才能下载模型
if embed_model_download_button and embed_model_type_selectbox == "huggingface":
    chroma_vectorstore_processor.download_model(
        repo_id=huggingface_repo_id_input,
        model_name_or_path=os.path.join(embedding_dir,embed_model_selectbox)
    )
    st.toast(i18n("Model downloaded successfully!"), icon="✅")


st.write(i18n("### Knowledge Base Setting"))


collection_choose_placeholder = st.empty()
with collection_choose_placeholder.container():
    st.write(i18n("Knowledge Base Choose"))
    collection_column, reinit_column = st.columns([0.7, 0.3])
    with collection_column:
        collection_choose = st.selectbox(
            label=i18n("Knowledge Base Choose"), 
            label_visibility="collapsed",
            options=chroma_vectorstore_processor.list_all_knowledgebase_collections(st.session_state.collection_counter),
            key="collection_choose"
        )
    with reinit_column:
        reinitialize_colleciton_button = st.button(
            label=i18n("Reinitialize Knowledge Base"),
            use_container_width=True,
            type="primary"
        )

embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")

if embed_model_type_selectbox == "openai":
    if not os.path.exists(embedding_config_file_path):
        # 创建 embedding_config.json
        with open(embedding_config_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    # 读取 embedding_config.json 中该名称的配置
    with open(embedding_config_file_path, "r", encoding="utf-8") as f:
        embedding_config = json.load(f)
    collection_config = embedding_config.get(collection_choose, {})
    
    # 创建 ChromaCollectionProcessor
    chroma_collection_processor = ChromaCollectionProcessor(
        collection_name=collection_choose,
        embedding_model_name_or_path=collection_config.get("embedding_model_name_or_path", embed_model_selectbox),
        embedding_model_type=collection_config.get("embedding_type", embed_model_type_selectbox),
        api_key=collection_config.get("api_key", os.getenv("AZURE_OAI_KEY",default="OPENAI_API_KEY")),
        base_url=collection_config.get("base_url", os.getenv("AZURE_OAI_ENDPOINT",default=None)),
        api_version=collection_config.get("api_version", os.getenv("API_VERSION",default="2024-02-15-preview")),
        api_type=collection_config.get("api_type", os.getenv("API_TYPE",default="openai")),   
    )

elif embed_model_type_selectbox == "huggingface":
    if not os.path.exists(embedding_config_file_path):
        # 创建 embedding_config.json
        with open(embedding_config_file_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(embedding_config_file_path, "r", encoding="utf-8") as f:
        embedding_config = json.load(f)
    collection_config = embedding_config.get(collection_choose, {})

    chroma_collection_processor = ChromaCollectionProcessor(
        collection_name=collection_choose,
        embedding_model_name_or_path=collection_config.get("embedding_model_name_or_path", embed_model_selectbox),
        embedding_model_type=collection_config.get("embedding_type", embed_model_type_selectbox),
    )

with st.expander(label=i18n("Collection Add/Delete"), expanded=False):
    collection_name_input = st.text_input(
        label=i18n("To be added Collection Name"),
        placeholder=i18n("Collection Name"),
        key="collection_name_input"
    )

    add_column, delete_column = st.columns(2)

    with add_column:
        add_collection_button = st.button(
            label=i18n("Add Collection"),
            use_container_width=True,
            type="primary"
        )
    
    with delete_column:
        delete_collection_button = st.button(
            label=i18n("Delete Collection"),
            use_container_width=True
        )



if add_collection_button:
    if st.session_state.collection_name_input != "":
        chroma_vectorstore_processor.create_knowledgebase_collection(collection_name=st.session_state.collection_name_input)
        st.session_state.collection_counter += 1
        st.session_state.document_counter += 1
        st.success(i18n("Collection added successfully."))
        chroma_vectorstore_processor.list_all_knowledgebase_collections(st.session_state.collection_counter)
        st.rerun()
    else:
        st.warning(i18n("Please enter the collection name."))


if delete_collection_button:
    chroma_vectorstore_processor.delete_knowledgebase_collection(collection_name=collection_choose)
    st.session_state.collection_counter += 1
    st.session_state.document_counter += 1
    st.success(i18n("Collection deleted successfully."))
    chroma_vectorstore_processor.list_all_knowledgebase_collections(st.session_state.collection_counter)
    st.rerun()

if reinitialize_colleciton_button:
    st.session_state.collection_counter += 1
    st.session_state.document_counter += 1
    chroma_vectorstore_processor.list_all_knowledgebase_collections(st.session_state.collection_counter)


st.write(i18n("### Choose files you want to embed"))

file_upload = st.file_uploader(
    label=i18n("Upload File"),
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=st.session_state["file_uploader_key"],
)

with st.expander(label=i18n("File Handling Configuration"), expanded=False):
    chunk_size_column, overlap_column = st.columns(2)
    with chunk_size_column:
        split_chunk_size = st.number_input(
            label=i18n("Chunk Size"),
            value=1000,
            step=1
        )
    with overlap_column:
        split_overlap = st.number_input(
            label=i18n("Overlap"),
            value=0,
            step=1
        )

upload_column,clear_column = st.columns([0.7,0.3])

with upload_column:
    upload_and_split_file_button = st.button(label=i18n("① Upload and Split Files"),use_container_width=True)

with clear_column:
    clear_file_button = st.button(label=i18n("Clear File"),use_container_width=True)
    if clear_file_button:
        st.session_state["file_uploader_key"] += 1
        st.session_state.pages = []
        st.rerun()

embed_button = st.button(label=i18n("② Embed Files"),use_container_width=True,type="primary")

st.write("")
st.write(i18n("Choose the file you want to preview"))

option_column,preview_column = st.columns([0.7,0.3])

with option_column:
    preview_placeholder = st.empty()

if file_upload:
    if upload_and_split_file_button:
        pages = []
        for file in file_upload:
            splitted_docs = text_split_execute(
                file=file,
                split_chunk_size=split_chunk_size,
                split_overlap=split_overlap,
            )

            pages.extend(splitted_docs)
            st.session_state.pages = pages

    # 优化预览文件列表的显示效果
    # 仅取每个文件的前50个字符作预览
    pages_option_preview = {f"{page.page_content[:50]}...": page.page_content for page in st.session_state.pages}

    option = preview_placeholder.selectbox(
        label=i18n("Choose the file you want to preview"),
        # label_visibility="collapsed",
        options=pages_option_preview.keys(),
        label_visibility="collapsed",
    )

    with preview_column:
        with st.popover(label=i18n("Content Preview"),use_container_width=True):
            with st.container(border=True):
                if option:
                    st.write(pages_option_preview[option])
else:
    file_upload.clear()
    preview_placeholder.empty()
    st.session_state.pages = []


if embed_button:
    if len(st.session_state.pages) == 0:
        st.warning(i18n("Please upload and split files first"))
    else:
        with st.spinner():
            chroma_collection_processor.add_documents(documents=st.session_state.pages)
            st.session_state.document_counter += 1
        st.toast("Embedding completed!")


st.write("---")
st.write(i18n("Knowledge Base ") + f"`{chroma_collection_processor.collection_name}`" + i18n(" 's files: "))

column1, column2 = st.columns([0.7,0.3])
with column1:
    file_names_inchroma = st.selectbox(
        label=i18n("Files in Knowledge Base"),
        label_visibility="collapsed",
        options=chroma_collection_processor.list_all_filechunks_metadata_name(st.session_state.document_counter)
    )
with column2:
    get_knowledge_base_info_button = st.button(label=i18n("Get Knowledge Base info"))

delete_file_button = st.button(label=i18n("Delete the File"),use_container_width=True)

if get_knowledge_base_info_button:
    with st.spinner(i18n("Getting file info...")):
        chroma_info_html = get_chroma_file_info(
            persist_path=KNOWLEDGE_BASE_PATH,
            collection_name=chroma_collection_processor.collection_name,
            file_name=file_names_inchroma,
            limit=len(chroma_collection_processor.list_collection_all_filechunks_content()),
            advance_info=False)
    components.html(chroma_info_html,
                    height=800)

if delete_file_button:
    chroma_collection_processor.delete_documents_from_same_metadata(files_name=file_names_inchroma)
    st.session_state.document_counter += 1
    st.rerun()