from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from markitdown import MarkItDown
from loguru import logger

import streamlit as st
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
from typing import List, Union, BinaryIO


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

_SUPPORTED_FILE_TYPES = ['pdf','md','txt','docx','doc','pptx','ppt','xlsx','xls','csv']


@st.cache_data
def choose_text_splitter(
    imput_stream: Union[List[BinaryIO],BinaryIO],
    chunk_size: int=1000,
    chunk_overlap: int=0
) -> List[Document]:
    '''
    根据文件类型选择不同的文本分割器
    
    Args:
        imput_stream: IO对象，单个或多个文件；
        chunk_size: 每个分块的大小，默认为1000；
        chunk_overlap: 每个分块的重叠部分，默认为0。
    '''

    #如果 imput_stream 是多个IO对象
    if isinstance(imput_stream,list):
        file_path_list = [i.name for i in imput_stream]
        splitted_docs = []

        for file in file_path_list:
            # 根据扩展名判断文件类型
            file_ext = os.path.splitext(file)[1].lstrip('.')
            if file_ext in _SUPPORTED_FILE_TYPES:
                try:
                    logger.info(f"Markitdown文件类型支持: {file_ext}")
                    md = MarkItDown()
                    res = md.convert(file)
                    document = Document(page_content=res,metadata={"source":file})
                    text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                    split_docs = text_splitter.split_documents(document)
                    splitted_docs.extend(split_docs)
                except Exception as e:
                    logger.error(f"Markitdown文件类型不支持: {file_ext}, 错误信息: {e}")
            else:
                try:
                    logger.info(f"Unstructured文件类型支持: {file_ext}")
                    loader = UnstructuredFileLoader(file)
                    document = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                    split_docs = text_splitter.split_documents(document)
                    splitted_docs.extend(split_docs)
                except Exception as e:
                    logger.error(f"Unstructured文件类型不支持: {file_ext}, 错误信息: {e}")
        
        return splitted_docs
    
    # 如果 imput_stream 是单个IO对象
    else:
        logger.info(f"object, 文件类型: {imput_stream.name}")
        file_ext = os.path.splitext(imput_stream.name)[1].lstrip('.')
        splitted_docs = []  # 初始化 splitted_docs 列表

        if file_ext in _SUPPORTED_FILE_TYPES:
            try:
                logger.info(f"Markitdown文件类型支持: {file_ext}")
                md = MarkItDown()
                res = md.convert(imput_stream.name)
                document = [Document(page_content=res.text_content,metadata={"source":imput_stream.name})]
                text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            except Exception as e:
                logger.error(f"Markitdown文件类型不支持: {file_ext}, 错误信息: {e}")
        else:
            try:
                logger.info(f"Unstructured文件类型支持: {file_ext}")
                loader = UnstructuredFileLoader(imput_stream.name)
                document = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(separators=_CHINESE_SEPARATORS,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            except Exception as e:
                logger.error(f"Unstructured文件类型不支持: {file_ext}, 错误信息: {e}")

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
    file: BinaryIO,
    split_chunk_size: int = 1000,
    split_overlap: int = 0,
) -> List[Document]:
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
    
        splitted_docs = choose_text_splitter(imput_stream=temp_file,chunk_size=split_chunk_size,chunk_overlap=split_overlap)
    # 手动删除临时文件
    os.remove(temp_file.name)
    # st.write(splitted_docs[0].page_content)
    # st.write(splitted_docs)

    return splitted_docs

@st.cache_data
def url_text_split_execute(
    url_content: dict,
    split_chunk_size: int = 1000,
    split_overlap: int = 0,
) -> List[Document]:
    """
    处理JinaReader返回的网页内容

    Args:
        url_content: JinaReader返回的网页内容，字典类型，只处理`content`字段；
        split_chunk_size: 每个分块的大小，默认为1000；
        split_overlap: 每个分块的重叠部分，默认为0。
    
    Returns:
        splitted_docs: 分割后的文档列表。
    """
    # 默认文件类型为Markdown
    file_suffix = ".md"
    file_name_without_suffix = url_content["content"].split("\n\n")[0]

    with tempfile.NamedTemporaryFile(prefix=file_name_without_suffix + "__",suffix=file_suffix,delete=False) as temp_file:
        temp_file.write(url_content["content"].encode("utf-8"))
        temp_file.seek(0)

        splitted_docs = choose_text_splitter(imput_stream=temp_file,chunk_size=split_chunk_size,chunk_overlap=split_overlap)
    # 手动删除临时文件
    os.remove(temp_file.name)

    return splitted_docs