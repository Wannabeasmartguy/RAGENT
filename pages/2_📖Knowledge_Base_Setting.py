import streamlit as st
import streamlit.components.v1 as components
import os
import json
import uuid
from configs.basic_config import I18nAuto, SUPPORTED_LANGUAGES, KNOWLEDGE_BASE_DIR
from configs.knowledge_base_config import ChromaVectorStoreProcessorWithNoApi, ChromaCollectionProcessorWithNoApi
from utils.chroma_utils import get_chroma_file_info
from utils.text_splitter.text_splitter_utils import text_split_execute
from utils.basic_utils import datetime_serializer
from model.config.embeddings import EmbeddingConfiguration, EmbeddingModelConfiguration, GlobalSettings, KnowledgeBaseConfiguration

# 全局变量和初始化
language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])
embedding_dir = "embeddings"
embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")

# 辅助函数
def init_session_state():
    if "embed_model_type" not in st.session_state:
        st.session_state.embed_model_type = "huggingface"
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "collection_counter" not in st.session_state:
        st.session_state.collection_counter = 0
    if "document_counter" not in st.session_state:
        st.session_state.document_counter = 0

def load_embedding_config():
    if not os.path.exists(embedding_config_file_path):
        initial_config = EmbeddingConfiguration(
            global_settings=GlobalSettings(default_model=""),
            models=[],
            knowledge_bases=[]
        )
        save_embedding_config(initial_config)
    
    with open(embedding_config_file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    return EmbeddingConfiguration(**config_data)

def save_embedding_config(config):
    with open(embedding_config_file_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, default=datetime_serializer)

def create_chroma_processors(embed_model_type, embed_model):
    """
    创建Chroma处理器, 根据embed_model_type和embed_model创建相应的处理器
    :param embed_model_type: 嵌入模型类型
    :param embed_model: 嵌入模型名称或路径
    :return: 返回Chroma向量存储处理器和Chroma集合处理器
    """
    if embed_model_type == "openai":
        chroma_vectorstore_processor = ChromaVectorStoreProcessorWithNoApi(
            embedding_model_name_or_path=embed_model,
            embedding_model_type=embed_model_type,
            api_key=os.getenv("AZURE_OAI_KEY", default="OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OAI_ENDPOINT", default=None),
            api_version=os.getenv("API_VERSION", default="2024-02-15-preview"),
            api_type=os.getenv("API_TYPE", default="openai"),
        )
    elif embed_model_type == "huggingface":
        chroma_vectorstore_processor = ChromaVectorStoreProcessorWithNoApi(
            embedding_model_name_or_path=os.path.join(embedding_dir, embed_model),
            embedding_model_type=embed_model_type,
        )
    
    embedding_config = load_embedding_config()
    model_id = get_or_create_model_id(embedding_config, embed_model_type, embed_model)
    
    chroma_collection_processor = ChromaCollectionProcessorWithNoApi(
        collection_name=st.session_state.get("collection_choose", ""),
        embedding_config=embedding_config,
        embedding_model_id=model_id
    )
    
    return chroma_vectorstore_processor, chroma_collection_processor

def get_or_create_model_id(embedding_config, embed_model_type, embed_model):
    current_model = next((model for model in embedding_config.models 
                          if model.embedding_type == embed_model_type 
                          and model.embedding_model_name_or_path == embed_model), None)
    
    if current_model:
        return current_model.id
    
    model_id = str(uuid.uuid4())
    if embed_model_type == "openai":
        new_model = EmbeddingModelConfiguration(
            id=model_id,
            name=f"OpenAI Embedding Model {model_id[:8]}",
            embedding_type="openai",
            embedding_model_name_or_path=embed_model,
            api_key=os.getenv("AZURE_OAI_KEY", default="OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OAI_ENDPOINT", default=None),
            api_version=os.getenv("API_VERSION", default="2024-02-15-preview"),
            api_type=os.getenv("API_TYPE", default="openai")
        )
    elif embed_model_type == "huggingface":
        new_model = EmbeddingModelConfiguration(
            id=model_id,
            name=f"Huggingface Embedding Model {model_id[:8]}",
            embedding_type="huggingface",
            embedding_model_name_or_path=embed_model
        )
    
    embedding_config.models.append(new_model)
    embedding_config.global_settings.default_model = model_id
    save_embedding_config(embedding_config)
    
    return model_id

# 侧边栏
def sidebar():
    with st.sidebar:
        current_directory = os.path.dirname(__file__)
        parent_directory = os.path.dirname(current_directory)
        logo_path = os.path.join(parent_directory, "img", "RAGenT_logo.png")
        logo_text = os.path.join(parent_directory, "img", "RAGenT_logo_with_text_horizon.png")
        st.logo(logo_text, icon_image=logo_path)

        st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
        st.write("---")
        st.info(i18n("Please select the embedded model when creating a knowledge base, as well as Reinitialize it once when switching knowledge bases."))

        embed_model_type = st.selectbox(
            label=i18n("Embed Model Type"),
            options=["openai", "huggingface"],
            key="embed_model_type",
        )
        # 定义embed_model_selector函数
        def embed_model_selector(embed_type):
            if embed_type == "openai":
                return ["text-embedding-ada-002"]
            elif embed_type == "huggingface":
                return ["bge-base-zh-v1.5", "bge-large-zh-v1.5"]
            else:
                return []

        embed_model = st.selectbox(
            label=i18n("Embed Model"),
            options=embed_model_selector(st.session_state.embed_model_type),
            key="embed_model",
        )

        with st.popover(label=i18n("Local embedding model download"), use_container_width=True):
            huggingface_repo_id = st.text_input(
                label=i18n("Huggingface repo id"),
                placeholder=i18n("Paste huggingface repo id here"),
            )
            if st.button(label=i18n("Download embedding model")) and embed_model_type == "huggingface":
                # 确保chroma_vectorstore_processor已定义
                if 'chroma_vectorstore_processor' not in locals():
                    from configs.knowledge_base_config import ChromaVectorStoreProcessor
                    chroma_vectorstore_processor = ChromaVectorStoreProcessor()
                
                chroma_vectorstore_processor.download_model(
                    repo_id=huggingface_repo_id,
                    model_name_or_path=os.path.join(embedding_dir, embed_model),
                )
                st.toast(i18n("Model downloaded successfully!"), icon="✅")

    return embed_model_type, embed_model

# 知识库设置
def knowledge_base_settings(chroma_vectorstore_processor, chroma_collection_processor):
    st.write(i18n("### Knowledge Base Setting"))

    with st.container():
        st.write(i18n("Knowledge Base Choose"))
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            collection_choose = st.selectbox(
                label=i18n("Knowledge Base Choose"),
                label_visibility="collapsed",
                options=chroma_vectorstore_processor.list_all_knowledgebase_collections(
                    st.session_state.collection_counter
                ),
                key="collection_choose",
            )
        with col2:
            if st.button(label=i18n("Reinitialize Knowledge Base"), use_container_width=True, type="primary"):
                st.session_state.collection_counter += 1
                st.session_state.document_counter += 1
                st.rerun()

    with st.expander(label=i18n("Collection Add/Delete"), expanded=False):
        collection_name = st.text_input(
            label=i18n("To be added Collection Name"),
            placeholder=i18n("Collection Name"),
            key="collection_name_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(label=i18n("Add Collection"), use_container_width=True, type="primary"):
                if collection_name:
                    chroma_vectorstore_processor.create_knowledgebase_collection(name=collection_name)
                    st.session_state.collection_counter += 1
                    st.session_state.document_counter += 1
                    st.success(i18n("Collection added successfully."))
                    st.rerun()

        with col2:
            if st.button(label=i18n("Delete Collection"), use_container_width=True):
                chroma_vectorstore_processor.delete_knowledgebase_collection(name=collection_name)
                st.session_state.collection_counter += 1
                st.session_state.document_counter += 1
                st.success(i18n("Collection deleted successfully."))
                st.rerun()

# 文件上传和处理
def file_upload_and_processing(chroma_collection_processor):
    st.write(i18n("### Choose files you want to embed"))

    file_upload = st.file_uploader(
        label=i18n("Upload File"),
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=st.session_state["file_uploader_key"],
    )

    with st.expander(label=i18n("File Handling Configuration"), expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            split_chunk_size = st.number_input(
                label=i18n("Chunk Size"),
                value=chroma_collection_processor.get_embedding_model_max_seq_len(),
                step=1,
            )
        with col2:
            split_overlap = st.number_input(label=i18n("Overlap"), value=0, step=1)

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        upload_and_split = st.button(label=i18n("① Upload and Split Files"), use_container_width=True)
    with col2:
        if st.button(label=i18n("Clear File"), use_container_width=True):
            st.session_state["file_uploader_key"] += 1
            st.session_state.pages = []
            st.rerun()

    embed_button = st.button(label=i18n("② Embed Files"), use_container_width=True, type="primary")

    if file_upload and upload_and_split:
        st.session_state.pages = []
        for file in file_upload:
            splitted_docs = text_split_execute(
                file=file,
                split_chunk_size=split_chunk_size,
                split_overlap=split_overlap,
            )
            st.session_state.pages.extend(splitted_docs)

    if st.session_state.pages:
        st.write(i18n("Choose the file you want to preview"))
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            pages_preview = {f"{page.page_content[:50]}...": page.page_content for page in st.session_state.pages}
            selected_page = st.selectbox(
                label=i18n("Choose the file you want to preview"),
                options=pages_preview.keys(),
                label_visibility="collapsed",
            )
        with col2:
            with st.popover(label=i18n("Content Preview"), use_container_width=True):
                if selected_page:
                    st.write(pages_preview[selected_page])

    if embed_button:
        if not st.session_state.pages:
            st.warning(i18n("Please upload and split files first"))
        else:
            with st.spinner():
                chroma_collection_processor.add_documents(documents=st.session_state.pages)
                st.session_state.document_counter += 1
            st.toast("Embedding completed!")

# 知识库内容管理
def knowledge_base_content_management(chroma_collection_processor):
    st.write("---")
    st.write(i18n("Knowledge Base ") + f"`{chroma_collection_processor.collection_name}`" + i18n(" 's files: "))

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        file_names = chroma_collection_processor.list_all_filechunks_metadata_name(st.session_state.document_counter)
        selected_file = st.selectbox(
            label=i18n("Files in Knowledge Base"),
            options=file_names,
            label_visibility="collapsed",
        )
    with col2:
        if st.button(label=i18n("Get Knowledge Base info")):
            with st.spinner(i18n("Getting file info...")):
                chroma_info_html = get_chroma_file_info(
                    persist_path=KNOWLEDGE_BASE_DIR,
                    collection_name=chroma_collection_processor.collection_name,
                    file_name=selected_file,
                    limit=len(chroma_collection_processor.list_collection_all_filechunks_content()),
                    advance_info=False,
                )
            components.html(chroma_info_html, height=800)

    if st.button(label=i18n("Delete the File"), use_container_width=True):
        chroma_collection_processor.delete_documents_from_same_metadata(files_name=selected_file)
        st.session_state.document_counter += 1
        st.rerun()

# 主函数
def main():
    init_session_state()
    embed_model_type, embed_model = sidebar()
    
    chroma_vectorstore_processor, chroma_collection_processor = create_chroma_processors(embed_model_type, embed_model)
    
    knowledge_base_settings(chroma_vectorstore_processor, chroma_collection_processor)
    file_upload_and_processing(chroma_collection_processor)
    knowledge_base_content_management(chroma_collection_processor)

if __name__ == "__main__":
    main()