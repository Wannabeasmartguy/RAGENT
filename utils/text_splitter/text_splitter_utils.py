from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownTextSplitter

import os

def choose_text_splitter(file_path,
                         chunk_size:int=1000,
                         chunk_overlap:int=0):
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
            if file_ext == '.pdf':
                loader = UnstructuredFileLoader(file_path.name)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            elif file_ext =='.md':
                loader = UnstructuredMarkdownLoader(file_path.name)
                document = loader.load()
                text_splitter = MarkdownTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            else:
                loader = UnstructuredFileLoader(file_path.name)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
        
        return splitted_docs
    
    # 如果 file_path 是一个obj
    else:
        file_ext = os.path.splitext(file_path.name)[1]
        if file_ext == '.pdf':
                loader = UnstructuredFileLoader(file_path.name)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
        elif file_ext =='.md':
            loader = UnstructuredMarkdownLoader(file_path.name)
            document = loader.load()
            text_splitter = MarkdownTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)
        else:
            loader = UnstructuredFileLoader(file_path.name)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)

        return split_docs