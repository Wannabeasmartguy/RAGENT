from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_core.documents import Document
from markitdown import MarkItDown
from loguru import logger

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import os
import tempfile
# 下载punkt分词器，避免在运行时报错
import nltk
try:
    nltk.data.find('punkt_tab')
    nltk.data.find('averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
from pathlib import Path
from typing import List


_CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
]


@st.cache_data
def choose_text_splitter(
    file_path,
    chunk_size: int=1000,
    chunk_overlap: int=0
):
    '''
    根据文件类型选择不同的文本分割器
    
    Args:
        file_path: 文件对象，单个临时文件或一个文件地址列表；
        chunk_size: 每个分块的大小，默认为1000；
        chunk_overlap: 每个分块的重叠部分，默认为0。
    '''

    #如果 file_path 是一个list
    if isinstance(file_path,list):
        file_path_list = [i.name for i in file_path]
        splitted_docs = []

        for file in file_path_list:
            # 根据扩展名判断文件类型
            file_ext = os.path.splitext(file)[1]
            if file_ext in ['.pdf','.md','.txt','.docx','.doc','.pptx','.ppt','.xlsx','.xls','.csv']:
                logger.info(f"Markitdown文件类型支持: {file_ext}")
                md = MarkItDown()
                res = md.convert(file)
                document = Document(page_content=res,metadata={"source":file})
                text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            else:
                logger.info(f"Unstructured文件类型支持: {file_ext}")
                loader = UnstructuredFileLoader(file)
                document = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
        
        return splitted_docs
    
    # 如果 file_path 是一个obj
    else:
        logger.info(f"object, 文件类型: {file_path.name}")
        file_ext = os.path.splitext(file_path.name)[1]
        splitted_docs = []  # 初始化 splitted_docs 列表

        if file_ext in ['.pdf','.md','.txt','.docx','.doc','.pptx','.ppt','.xlsx','.xls','.csv']:
            logger.info(f"Markitdown文件类型支持: {file_ext}")
            md = MarkItDown()
            res = md.convert(file_path.name)
            document = [Document(page_content=res.text_content,metadata={"source":file_path.name})]
            text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)
            splitted_docs.extend(split_docs)
        else:
            logger.info(f"Unstructured文件类型支持: {file_ext}")
            loader = UnstructuredFileLoader(file_path.name)
            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)
            splitted_docs.extend(split_docs)

        return splitted_docs
    

def simplify_filename(original_name):
    """
    Simplify a given filename by removing additional characters and keeping the base name and extension.

    Parameters:
    - original_name (str): The original name of the file.

    Returns:
    - new_name (str): The simplified name of the file.
    """
    # Split the original name by '__' and take the first part
    base_name = original_name.split('__')[0]
    # Extract the extension
    extension = original_name.split('.')[-1]
    # Construct the new file name
    new_name = f"{base_name}.{extension}"
    
    return new_name


@st.cache_data
def text_split_execute(
    file: UploadedFile,
    split_chunk_size: int = 1000,
    split_overlap: int = 0,
):
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

    return splitted_docs

def url_text_split_execute(
    url_content: dict,
    split_chunk_size: int = 1000,
    split_overlap: int = 0,
):
    import time
    # 默认文件类型为Markdown
    file_suffix = ".md"
    file_name_without_suffix = url_content["content"].split("\n\n")[0]

    with tempfile.NamedTemporaryFile(prefix=file_name_without_suffix + "__",suffix=file_suffix,delete=False) as temp_file:
        temp_file.write(url_content["content"].encode("utf-8"))
        temp_file.seek(0)

        splitted_docs = choose_text_splitter(file_path=temp_file,chunk_size=split_chunk_size,chunk_overlap=split_overlap)
    # 手动删除临时文件
    os.remove(temp_file.name)

    return splitted_docs