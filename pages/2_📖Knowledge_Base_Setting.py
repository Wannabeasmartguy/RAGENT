import streamlit as st
import streamlit.components.v1 as components

from pathlib import Path
import os
import tempfile

from configs.basic_config import I18nAuto,SUPPORTED_LANGUAGES
from configs.knowledge_base_config import ChromaCollectionProcessor, ChromaVectorStoreProcessor
from utils.chroma_utils import *
from utils.text_splitter.text_splitter_utils import *


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])


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

with st.sidebar:
    # 获得同级文件夹 /img 的路径
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, 'img', 'RAGenT_logo.png')
    st.image(logo_path)

    st.page_link("pages/1_🤖AgentChat.py", label="🤖 AgentChat")
    st.write("---")

    embed_model_type_selectbox = st.selectbox(label=i18n("Embed Model Type"),
                                              options=["openai", "huggingface"],
                                              key="embed_model_type")
    embed_model_selectbox = st.selectbox(label=i18n("Embed Model"),
                                         options=embed_model_selector(st.session_state.embed_model_type),
                                         key="embed_model")
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
        embedding_model_name_or_path="embedding model/" + embed_model_selectbox,
        embedding_model_type=embed_model_type_selectbox,
    )


collection_choose_placeholder = st.empty()
with collection_choose_placeholder.container():
    collection_choose = st.selectbox(
        label=i18n("Knowledge Base Choose"), 
        #  label_visibility="collapsed",
        options=chroma_vectorstore_processor.list_all_knowledgebase_collections(),
        key="collection_choose"
    )

if embed_model_type_selectbox == "openai":
    chroma_collection_processor = ChromaCollectionProcessor(
        collection_name=collection_choose,
        embedding_model_name_or_path=embed_model_selectbox,
        embedding_model_type=embed_model_type_selectbox,
        api_key=os.getenv("AZURE_OAI_KEY",default="OPENAI_API_KEY"),
        base_url=os.getenv("AZURE_OAI_ENDPOINT",default=None),
        api_version=os.getenv("API_VERSION",default="2024-02-15-preview"),
        api_type=os.getenv("API_TYPE",default="openai"),
    )
elif embed_model_type_selectbox == "huggingface":
    chroma_collection_processor = ChromaCollectionProcessor(
        collection_name=collection_choose,
        embedding_model_name_or_path="embedding model/" + embed_model_selectbox,
        embedding_model_type=embed_model_type_selectbox
    )

with st.expander(label=i18n("Collection Add/Delete"), expanded=False):
    collection_name_input = st.text_input(
        label=i18n("To be added Collection Name"),
        placeholder=i18n("Collection Name"),
        key="collection_name_input"
    )
    add_collection_button = st.button(
        label=i18n("Add Collection"),
        use_container_width=True
    )


reinitialize_colleciton_button = st.button(
    label=i18n("Reinitialize Knowledge Base"),
)

delete_collection_button = st.button(
    label=i18n("Delete Collection"),
    use_container_width=True
)

if add_collection_button:
    if st.session_state.collection_name_input != "":
        chroma_vectorstore_processor.create_knowledgebase_collection(collection_name=st.session_state.collection_name_input)
        st.success(i18n("Collection added successfully."))
        chroma_vectorstore_processor.list_all_knowledgebase_collections()
        st.rerun()
    else:
        st.warning(i18n("Please enter the collection name."))


if delete_collection_button:
    chroma_collection_processor.delete_knowledgebase_collection()
    st.success(i18n("Collection deleted successfully."))
    chroma_vectorstore_processor.list_all_knowledgebase_collections()

if reinitialize_colleciton_button:
    chroma_vectorstore_processor.list_all_knowledgebase_collections()


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
            chroma_collection_processor.add_documents(documents=st.session_state.pages)
        st.toast("Embedding completed!")


st.write("---")
st.write(f"知识库 `{chroma_collection_processor.collection_name}` 中的文件：")

column1, column2 = st.columns([0.7,0.3])
with column1:
    file_names_inchroma = st.selectbox(label=i18n("Files in Knowledge Base"),
                                       label_visibility="collapsed",
                                       options=chroma_collection_processor.list_all_filechunks_metadata_name())
with column2:
    get_knowledge_base_info_button = st.button(label=i18n("Get Knowledge Base info"))

delete_file_button = st.button(label=i18n("Delete the File"),
                               use_container_width=True)

if get_knowledge_base_info_button:
    chroma_info_html = get_chroma_file_info(
        persist_path="./knowledgebase",
        collection_name=chroma_collection_processor.collection_name,
        file_name=file_names_inchroma,
        limit=len(chroma_collection_processor.list_collection_all_filechunks_content()),
        advance_info=False)
    components.html(chroma_info_html,
                    height=800)

if delete_file_button:
    chroma_collection_processor.delete_documents_from_same_metadata(files_name=file_names_inchroma)
    st.rerun()