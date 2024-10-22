import streamlit as st
import streamlit.components.v1 as components
import os
import json
import uuid
from typing import Dict, List
from loguru import logger
from core.basic_config import (
    I18nAuto,
    set_pages_configs_in_common,
    SUPPORTED_LANGUAGES,
    KNOWLEDGE_BASE_DIR,
)
from core.kb_processors import (
    ChromaVectorStoreProcessorWithNoApi,
    ChromaCollectionProcessorWithNoApi,
)
from utils.chroma_utils import get_chroma_file_info
from utils.text_splitter.text_splitter_utils import (
    text_split_execute, 
    url_text_split_execute
)
from utils.basic_utils import datetime_serializer
from model.config.embeddings import (
    EmbeddingConfiguration,
    EmbeddingModelConfiguration,
    GlobalSettings,
    KnowledgeBaseConfiguration,
)

# 全局变量声明
global chroma_vectorstore_processor, chroma_collection_processor
chroma_vectorstore_processor = None
chroma_collection_processor = None

# 全局变量和初始化
language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(language=SUPPORTED_LANGUAGES[language])
embedding_dir = "embeddings"
embedding_config_file_path = os.path.join("dynamic_configs", "embedding_config.json")


# 更新加载embed model配置文件的函数
def load_embedding_model_config() -> Dict[str, Dict[str, List[str]]]:
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "embedding_model_config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# 辅助函数
def init_session_state():
    if "embed_model_type" not in st.session_state:
        st.session_state.embed_model_type = "sentence_transformer"
    if "pages" not in st.session_state:
        st.session_state.pages = []
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "collection_counter" not in st.session_state:
        st.session_state.collection_counter = 0
    if "document_counter" not in st.session_state:
        st.session_state.document_counter = 0
    if "collection_name_check" not in st.session_state:
        # 用于检查collection name是否合法，初始合法以启用add collection按钮
        st.session_state.collection_name_check = True
    if "create_collection_expander_state" not in st.session_state:
        # 用于控制collection add/delete expander的展开状态，初始不展开
        st.session_state.create_collection_expander_state = False
    if "url_scrape_result" not in st.session_state:
        st.session_state.url_scrape_result = None


def load_embedding_config():
    if not os.path.exists(embedding_config_file_path):
        initial_config = EmbeddingConfiguration(
            global_settings=GlobalSettings(default_model=""),
            models=[],
            knowledge_bases=[],
        )
        save_embedding_config(initial_config)

    with open(embedding_config_file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    return EmbeddingConfiguration(**config_data)


def save_embedding_config(config):
    with open(embedding_config_file_path, "w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2, default=datetime_serializer)


def create_chroma_vectorstore_processor(embed_model_type, embed_model):
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
            api_key=os.getenv("OPENAI_API_KEY", default="OPENAI_API_KEY"),
        )
    elif embed_model_type == "aoai":
        chroma_vectorstore_processor = ChromaVectorStoreProcessorWithNoApi(
            embedding_model_name_or_path=embed_model,
            embedding_model_type=embed_model_type,
            api_key=os.getenv("AZURE_OAI_KEY", default="OPENAI_API_KEY"),
            base_url=os.getenv("AZURE_OAI_ENDPOINT", default=None),
            api_version=os.getenv("API_VERSION", default="2024-02-15-preview"),
            api_type=os.getenv("API_TYPE", default="azure"),
        )
    elif embed_model_type == "sentence_transformer":
        chroma_vectorstore_processor = ChromaVectorStoreProcessorWithNoApi(
            embedding_model_name_or_path=embed_model,
            embedding_model_type=embed_model_type,
        )

    return chroma_vectorstore_processor


def create_chroma_collection_processor():
    embedding_config = load_embedding_config()
    model_id = get_or_create_model_id(
        st.session_state.get("collection_choose"), embedding_config
    )

    chroma_collection_processor = ChromaCollectionProcessorWithNoApi(
        collection_name=st.session_state.get("collection_choose"),
        embedding_config=embedding_config,
        embedding_model_id=model_id,
    )

    return chroma_collection_processor


# 创建Chroma处理器
def create_chroma_processors():
    global chroma_vectorstore_processor, chroma_collection_processor
    chroma_vectorstore_processor = create_chroma_vectorstore_processor(
        embed_model_type, embed_model
    )

    # 如果没有collection,创建一个默认的
    if not chroma_vectorstore_processor.knowledgebase_collections:
        logger.info("No collection, create a default collection.")
        chroma_vectorstore_processor.create_knowledgebase_collection(
            collection_name="default"
        )
        st.session_state.collection_choose = "default"
        st.toast(
            i18n(
                "First time creation of knowledge base, create a new default collection."
            ),
            icon="🔔",
        )

    # 确保collection_choose存在且有效
    if (
        "collection_choose" not in st.session_state
        or st.session_state.collection_choose
        not in chroma_vectorstore_processor.knowledgebase_collections
    ):
        st.session_state.collection_choose = next(
            iter(chroma_vectorstore_processor.knowledgebase_collections)
        )

    chroma_collection_processor = create_chroma_collection_processor()
    return chroma_vectorstore_processor, chroma_collection_processor


def get_or_create_model_id(collection_name, embedding_config):
    # 先从embedding_config中获得collection_name对应的embedding_model_id
    current_collection = next(
        (
            collection
            for collection in embedding_config.knowledge_bases
            if collection.name == collection_name
        ),
        None,
    )

    if current_collection:
        return current_collection.embedding_model_id

    model_id = str(uuid.uuid4())
    if embed_model_type == "openai":
        new_model = EmbeddingModelConfiguration(
            id=model_id,
            name=f"OpenAI Embedding Model {model_id[:8]}",
            embedding_type="openai",
            embedding_model_name_or_path=embed_model,
            api_key=os.getenv("OPENAI_API_KEY", default="OPENAI_API_KEY"),
        )
    elif embed_model_type == "aoai":
        new_model = EmbeddingModelConfiguration(
            id=model_id,
            name=f"Azure OpenAI Embedding Model {model_id[:8]}",
            embedding_type="aoai",
            embedding_model_name_or_path=embed_model,
            api_key=os.getenv("AZURE_OAI_KEY"),
            base_url=os.getenv("AZURE_OAI_ENDPOINT", default=None),
            api_version=os.getenv("API_VERSION", default="2024-02-15-preview"),
            api_type=os.getenv("API_TYPE", default="azure"),
        )
    elif embed_model_type == "sentence_transformer":
        new_model = EmbeddingModelConfiguration(
            id=model_id,
            name=f"Huggingface (SentenceTransformer) Embedding Model {model_id[:8]}",
            embedding_type="sentence_transformer",
            embedding_model_name_or_path=embed_model,
        )

    embedding_config.models.append(new_model)
    embedding_config.global_settings.default_model = model_id
    save_embedding_config(embedding_config)

    return model_id


try:
    VERSION = "0.1.1"
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, "img", "RAGenT_logo.png")
    set_pages_configs_in_common(
        version=VERSION, title="Knowledge Base Management", page_icon_path=logo_path
    )
except:
    st.rerun()

init_session_state()

# 侧边栏
with st.sidebar:
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.dirname(current_directory)
    logo_path = os.path.join(parent_directory, "img", "RAGenT_logo.png")
    logo_text = os.path.join(
        parent_directory, "img", "RAGenT_logo_with_text_horizon.png"
    )
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.write("---")
    st.info(
        i18n(
            "Please select the embedded model when creating a knowledge base, as well as Reinitialize it once when switching knowledge bases."
        )
    )

    # 获取embed model的配置
    embedding_config = load_embedding_model_config()

    embed_model_type = st.selectbox(
        label=i18n("Embed Model Type"),
        options=list(embedding_config.keys()),
        key="embed_model_type",
        format_func=lambda x: embedding_config[x].get("display_name", x),
    )

    if embed_model_type == "sentence_transformer":
        embed_model_owner_organization = st.selectbox(
            label=i18n("Embed Model Owner Organization"),
            options=list(
                embedding_config["sentence_transformer"]["owner_organization"]
            ),
            key="embed_model_owner_organization",
        )

        embed_model = st.selectbox(
            label=i18n("Embed Model"),
            options=embedding_config["sentence_transformer"]["owner_organization"][
                st.session_state.embed_model_owner_organization
            ],
            key="embed_model",
        )
    else:
        embed_model = st.selectbox(
            label=i18n("Embed Model"),
            options=embedding_config[embed_model_type]["models"],
            key="embed_model",
        )

    if st.session_state.embed_model_type == "sentence_transformer":
        with st.expander(label=i18n("Local embedding model download"), expanded=False):
            if st.button(
                label=i18n("Download embedding model"), 
                use_container_width=True
            ):
                # huggingface repo id就是organization+model_name
                ChromaVectorStoreProcessorWithNoApi.download_model(
                    repo_id=f"{st.session_state.embed_model_owner_organization}/{st.session_state.embed_model}",
                    model_name_or_path=os.path.join(
                        embedding_dir, st.session_state.embed_model
                    ),
                )
                st.toast(i18n("Model downloaded successfully!"), icon="✅")

# 创建Chroma处理器
chroma_vectorstore_processor, chroma_collection_processor = create_chroma_processors()

# 知识库设置
st.write(i18n("### Knowledge Base Setting"))

with st.container():
    st.write(i18n("Knowledge Base Choose"))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:

        def collection_change():
            st.session_state.document_counter += 1
            global chroma_vectorstore_processor, chroma_collection_processor
            chroma_vectorstore_processor, chroma_collection_processor = (
                create_chroma_processors()
            )
            logger.info(f"Selected collection: {st.session_state.collection_choose}")

        collection_choose = st.selectbox(
            label=i18n("Knowledge Base Choose"),
            label_visibility="collapsed",
            options=chroma_vectorstore_processor.knowledgebase_collections,
            key="collection_choose",
            on_change=collection_change,
        )
    with col2:

        def reinitialize_knowledge_base():
            st.session_state.collection_counter += 1
            st.session_state.document_counter += 1
            global chroma_vectorstore_processor, chroma_collection_processor
            chroma_vectorstore_processor = create_chroma_vectorstore_processor(
                embed_model_type, embed_model
            )
            # chroma_collection_processor = create_chroma_collection_processor()

        st.button(
            label=i18n("Reinitialize Knowledge Base"),
            use_container_width=True,
            type="primary",
            on_click=reinitialize_knowledge_base,
        )

with st.expander(
    label=i18n("Collection Add/Delete"),
    expanded=st.session_state.create_collection_expander_state,
):

    def add_collection_callback():
        # 更新expander状态
        st.session_state.create_collection_expander_state = True
        # 检查collection name是否合法
        if st.session_state.collection_name_check:
            chroma_vectorstore_processor.create_knowledgebase_collection(
                collection_name=st.session_state.create_collection_name_input
            )
            st.session_state.create_collection_name_input = ""
            st.session_state.collection_counter += 1
            st.session_state.document_counter += 1
            st.success(i18n("Collection added successfully."))
            st.session_state.create_collection_expander_state = False
        else:
            st.error(i18n("Collection name is invalid."))

    def delete_collection_callback():
        global chroma_vectorstore_processor, chroma_collection_processor
        # 更新expander状态
        st.session_state.create_collection_expander_state = True
        # 删除collection
        chroma_vectorstore_processor.delete_knowledgebase_collection(
            collection_name=st.session_state.delete_collection_name_selectbox
        )
        st.session_state.collection_counter += 1
        st.session_state.document_counter += 1
        st.success(i18n("Collection deleted successfully."))

        # 如果删除后，vectorstore中没有collection，则创建新的默认collection
        if not chroma_vectorstore_processor.knowledgebase_collections:
            chroma_vectorstore_processor.create_knowledgebase_collection(
                collection_name="default"
            )
            # 并提醒
            st.toast(
                i18n("No collection left, create a new default collection."), icon="🔔"
            )

        # 更新collection_choose
        chroma_vectorstore_processor, chroma_collection_processor = (
            create_chroma_processors()
        )

    add_collection_tab, delete_collection_tab = st.tabs(
        [i18n("Add Collection"), i18n("Delete Collection")]
    )
    with add_collection_tab:

        def collection_name_change():
            """
            检查collection name是否合法

            1. 名称的长度必须介于 1 到 63 个字符之间。
            2. 名称只能包含Unicode字符、数字、点、短划线和下划线。
            3. 名称不得包含两个连续的点。
            4. 名称不得包含两个连续的下划线。
            5. 名称不得为纯数字。
            6. 名称不得重复
            """
            # 更新expander状态
            st.session_state.create_collection_expander_state = True
            # 检查collection name是否合法
            collection_name = st.session_state.create_collection_name_input
            if len(collection_name) < 1 or len(collection_name) > 63:
                st.error(
                    i18n(
                        "Knowledge Base name must be between 1 and 63 characters long, containing only Unicode characters, numbers, dots, hyphens, and underscores."
                    )
                )
                st.session_state.collection_name_check = False
                return False
            if not all(c.isalnum() or c in ".-_" for c in collection_name):
                st.error(
                    i18n(
                        "Knowledge Base name can only contain Unicode characters, numbers, dots, hyphens, and underscores."
                    )
                )
                st.session_state.collection_name_check = False
                return False
            if ".." in collection_name:
                st.error(
                    i18n("Knowledge Base name must not contain two consecutive dots.")
                )
                st.session_state.collection_name_check = False
                return False
            if "__" in collection_name:
                st.error(
                    i18n(
                        "Knowledge Base name must not contain two consecutive underscores."
                    )
                )
                st.session_state.collection_name_check = False
                return False
            if collection_name.isdigit():
                st.error(i18n("Knowledge Base name must not be a pure number."))
                st.session_state.collection_name_check = False
                return False
            if (
                collection_name
                in chroma_vectorstore_processor.knowledgebase_collections
            ):
                st.error(i18n("Knowledge Base name must not be repeated."))
                st.session_state.collection_name_check = False
                return False

            st.session_state.collection_name_check = True
            return True

        create_collection_name_input = st.text_input(
            label=i18n("To be added Collection Name"),
            placeholder=i18n("Collection Name"),
            key="create_collection_name_input",
            on_change=collection_name_change,
        )
        st.button(
            label=i18n("Add Collection"),
            use_container_width=True,
            type="primary",
            on_click=add_collection_callback,
            disabled=not st.session_state.collection_name_check,
        )

    with delete_collection_tab:
        delete_collection_name_selectbox = st.selectbox(
            label=i18n("To be deleted Collection Name"),
            options=chroma_vectorstore_processor.knowledgebase_collections,
            key="delete_collection_name_selectbox",
        )
        st.button(
            label=i18n("Delete Collection"),
            use_container_width=True,
            on_click=delete_collection_callback,
        )

# 文件上传和处理
st.write(i18n("### Choose files you want to embed"))

with st.container(border=True):
    upload_local_file_tab, upload_url_tab = st.tabs(
        [i18n("Upload Local File"), i18n("Upload URL")]
    )

with upload_local_file_tab:
    file_upload = st.file_uploader(
        label=i18n("Upload File"),
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=st.session_state["file_uploader_key"],
    )

with upload_url_tab:
    url_input = st.text_input(
        label=i18n("URL"),
        placeholder=i18n("Paste blog or article URL here."),
        help=i18n("Only paste one URL at a time."),
        key="url_input",
    )

    parse_url_button = st.button(
        label=i18n("Parse URL"),
        use_container_width=True,
        type="primary",
    )

    if parse_url_button:
        if url_input:
            from modules.scraper.url import JinaScraper
            scraper = JinaScraper()
            url_scrape_result = scraper.scrape(url_input)

            if url_scrape_result["status_code"] == 200:
                st.session_state.url_scrape_result = url_scrape_result
            else:
                st.error(i18n("Failed to parse URL."))
        else:
            st.toast(i18n("Please enter a URL."), icon="🚨")
    
    if st.session_state.url_scrape_result:
        with st.expander(label=i18n("Content Preview"), expanded=False):
            st.write(st.session_state.url_scrape_result["content"])

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
    upload_and_split = st.button(
        label=i18n("① Upload and Split Files"), use_container_width=True
    )
with col2:
    if st.button(label=i18n("Clear File"), use_container_width=True):
        st.session_state["file_uploader_key"] += 1
        st.session_state.url_scrape_result = None
        st.session_state.pages = []
        st.rerun()

embed_button = st.button(
    label=i18n("② Embed Files"), use_container_width=True, type="primary"
)

if file_upload and upload_and_split:
    st.session_state.pages = []
    for file in file_upload:
        try:
            splitted_docs = text_split_execute(
                file=file,
                split_chunk_size=split_chunk_size,
                split_overlap=split_overlap,
            )
            st.session_state.pages.extend(splitted_docs)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
elif st.session_state.url_scrape_result and upload_and_split:
    st.session_state.pages = []
    try:
        splitted_docs = url_text_split_execute(
            url_content=st.session_state.url_scrape_result,
            split_chunk_size=split_chunk_size,
            split_overlap=split_overlap,
        )
        st.session_state.pages.extend(splitted_docs)
    except Exception as e:
        st.error(f"Error processing URL content: {str(e)}")
elif not st.session_state.pages and not st.session_state.url_scrape_result and upload_and_split:
    st.toast(i18n("Please upload files or parse a URL first"), icon="🚨")

if st.session_state.pages:
    st.write(i18n("Choose the file you want to preview"))
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        pages_preview = {
            f"{page.page_content[:50]}...": page.page_content
            for page in st.session_state.pages
        }
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
            try:
                # 始终使用当前知识库的处理器
                chroma_collection_processor.add_documents(documents=st.session_state.pages)
                st.session_state.document_counter += 1
                st.toast(i18n("嵌入完成！"), icon="✅")
                # 清空st.session_state.pages
                st.session_state.pages = []
                # 清空url_scrape_result和url_input
                st.session_state.url_scrape_result = None
                st.session_state.url_input = ""
            except Exception as e:
                st.error(f"Error embedding files: {str(e)}")

# 知识库内容管理
st.write(i18n("### Knowledge Base Content Management"))
st.write(
    i18n("Knowledge Base ")
    + f"`{chroma_collection_processor.collection_name}`"
    + i18n(" 's files: ")
)

col1, col2 = st.columns([0.7, 0.3])
with col1:
    try:
        file_names = chroma_collection_processor.list_all_filechunks_metadata_name(
            st.session_state.document_counter
        )
        selected_file = st.selectbox(
            label=i18n("Files in Knowledge Base"),
            options=file_names,
            label_visibility="collapsed",
        )
    except Exception as e:
        st.error(f"Error getting file list: {str(e)}")
        selected_file = None
with col2:
    get_knowledge_base_info_button = st.button(label=i18n("Get Knowledge Base info"))
if get_knowledge_base_info_button:
    with st.spinner(i18n("Getting file info...")):
        try:
            chroma_info_html = get_chroma_file_info(
                persist_path=KNOWLEDGE_BASE_DIR,
                collection_name=chroma_collection_processor.collection_id,
                file_name=selected_file,
                limit=len(
                    chroma_collection_processor.list_collection_all_filechunks_content()
                ),
                advance_info=False,
            )
            st.html(chroma_info_html)
        except Exception as e:
            st.error(f"Error getting knowledge base info: {str(e)}")

if st.button(label=i18n("Delete File"), use_container_width=True):
    if selected_file:
        try:
            chroma_collection_processor.delete_documents_from_same_metadata(
                files_name=selected_file
            )
            st.session_state.document_counter += 1
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting file: {str(e)}")
    else:
        st.warning(i18n("Please select the file you want to delete"))
