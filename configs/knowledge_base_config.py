import os
import json

from typing import Literal

import streamlit as st

from huggingface_hub import snapshot_download
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import chroma

from utils.text_splitter.text_splitter_utils import simplify_filename

from configs.basic_config import I18nAuto

i18n = I18nAuto()


def create_vectorstore(persist_vec_path:str,
                       embedding_model_type:Literal['OpenAI','Hugging Face(local)'],
                       embedding_model:str):
    '''
    Create vectorstore.
    '''
    if persist_vec_path == "":
        raise "Please provide a path to persist the vectorstore."
    
    local_embedding_model = 'embedding model/'+embedding_model

    import os
    if os.path.isabs(persist_vec_path):
        if embedding_model_type == 'OpenAI':
            embeddings = AzureOpenAIEmbeddings(
                                                openai_api_type=os.getenv('API_TYPE'),
                                                azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                                                openai_api_key=os.getenv('AZURE_OAI_KEY'),
                                                openai_api_version=os.getenv('API_VERSION'),
                                                azure_deployment="text-embedding-ada-002",
                                                )
        elif embedding_model_type == 'Hugging Face(local)':
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model)
            except:
                # 如果 embedding model 的前三个字母是 bge ,则在 repo_id 前加上 BAAI/
                if embedding_model[:3] == 'bge':
                    st.toast("Downloading embedding model, please wait...")
                    try:
                        snapshot_download(repo_id="BAAI/"+embedding_model,
                                        local_dir=local_embedding_model)
                        st.toast("Download complete.")
                    except:
                        st.toast("Download failed, please check your network connection and try again.")
                    embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model)

        # global vectorstore
        vectorstore = chroma.Chroma(persist_directory=persist_vec_path,embedding_function=embeddings)
        vectorstore.persist()
    else:
        raise "The path is not valid."
    
    return vectorstore


class KnowledgeBase:
    '''
    用于管理本地知识库的基类，记录所有子知识库的embedding信息
    '''
    def __init__(self):
        '''
        初始化时，先读取本地的`embedding_config.json`,该json的结构如下
        {
            "Knowledge_base_a": {
                "embedding_model_type": "OpenAI",
                "embedding_model": "text-embedding-ada-002"
            },
            "Knowledge_base_b":{
                "embedding_model_type": "Hugging Face(local)",
                "embedding_model": "bge-base-zh-v1.5"
            }
        }
        '''
        if os.path.exists("embedding_config.json"):
            with open("embedding_config.json", "r", encoding='utf-8') as f:
                self.__embedding_config = json.load(f)
        else:
            # 如果不存在"embedding_config.json"，则创建它
            self.__embedding_config = {
                "default_empty_vec": {
                    "embedding_model_type": "OpenAI",
                    "embedding_model": "text-embedding-ada-002"
                }
            }
            with open("embedding_config.json", 'w', encoding='utf-8') as file:
                json.dump(self.__embedding_config, file, ensure_ascii=False, indent=4)
            tmp_vec_path = os.path.join(os.getcwd(), "knowledge base", "default_empty_vec")
            if os.path.exists(tmp_vec_path) is False:
                os.makedirs(tmp_vec_path)
            create_vectorstore(tmp_vec_path, "OpenAI", "text-embedding-ada-002")
                
        self.knowledge_bases = list(self.__embedding_config.keys())

    def reinitialize(self):
        '''重新初始化，以重载json中内容'''
        self.__init__()

    def get_embedding_model(self, knowledge_base_name:str):
        """
        根据知识库名称获取嵌入模型的分类和名称

        Args:
            knowledge_base_name: `embedding_config.json` 中保存的名称;
        
        Return: 
            `embedding_model_type`: str,`embedding_model`: str
        """
        
        if knowledge_base_name in self.knowledge_bases:
            return self.__embedding_config[knowledge_base_name]["embedding_model_type"],self.__embedding_config[knowledge_base_name]["embedding_model"]
        else:
            raise ValueError(f"未找到名为{knowledge_base_name}的知识库")
        
    def get_persist_vec_path(self, knowledge_base_name:str):
        '''在默认路径下按名字查找知识库，并返回知识库的路径'''
        vec_root_path = os.path.join(os.getcwd(), "knowledge base")
        vec_path = os.path.join(vec_root_path, knowledge_base_name)
        if os.path.exists(vec_path):
            return vec_path
        else:
            raise ValueError(f"未找到名为{knowledge_base_name}的知识库")
        

class SubKnowledgeBase(KnowledgeBase):
    '''
    用于管理子知识库的基类
    '''
    def __init__(self, knowledge_base_name:str):
        super().__init__()
        self.knowledge_base_name = knowledge_base_name
        self.embedding_model_type,self.embedding_model = self.get_embedding_model(knowledge_base_name)
        self.persist_vec_path = self.get_persist_vec_path(knowledge_base_name)
        self.vectorstore = self.vectorstore_init_create(persist_vec_path=self.persist_vec_path,
                                                        embedding_model_type=self.embedding_model_type,
                                                        local_embedding_model=self.embedding_model)
    
    def reinitialize(self,knowledge_base_name):
        '''重新初始化，以重载json中内容'''
        self.__init__(knowledge_base_name=knowledge_base_name)

    def vectorstore_init_create(self,persist_vec_path, 
                                embedding_model_type,
                                local_embedding_model):
        '''
        Generic Vectorstore Creation Functions

        Args:
            persist_vec_path: chroma persist path.
            embedding_model_type: `OpenAI` or `Hugging Face(local)`
            local_embedding_model: If `embedding_model_type` == `OpenAI`, there's only one model, here will give a specific embedding model. 
            progress: create gradio progress bar.
        '''
        local_embedding_model_path = os.path.join("embedding model/",local_embedding_model)
        if embedding_model_type == 'Hugging Face(local)':
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model_path)
                vectorstore = chroma.Chroma(
                    persist_directory=persist_vec_path, embedding_function=embeddings
                )
            except:
                st.toast("Downloading embedding model...")
                if local_embedding_model[:3] == 'bge':
                    snapshot_download(repo_id="BAAI/"+local_embedding_model,
                                        local_dir=local_embedding_model_path)
                    st.toast("Download complete.")
                    embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model_path)
                    vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                                embedding_function=embeddings)
                    
        elif embedding_model_type == 'OpenAI':
            vectorstore = chroma.Chroma(persist_directory=persist_vec_path, 
                                embedding_function=AzureOpenAIEmbeddings(
                                openai_api_type=os.getenv('API_TYPE'),
                                azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                                openai_api_key=os.getenv('AZURE_OAI_KEY'),
                                openai_api_version=os.getenv('API_VERSION'),
                                azure_deployment="text-embedding-ada-002",
                                ))
        return vectorstore


    def add_file_in_vectorstore(
            self,
            split_docs:list,
            file_obj,   # get it from 'file' (gr.file)
        ):
        '''
        Add file to vectorstore.
        '''
        if file_obj == None:
            raise "You haven't chosen a file yet."

        if isinstance(file_obj, list):
            self.vectorstore.add_documents(documents=split_docs)


    def delete_flie_in_vectorstore(self,files_samename:str):
        '''
        Get the file's ids first, then delete by vector IDs.
        '''
        # Specify the target file
        try:
            metadata = self.vectorstore.get()
        except NameError as n:
            raise 'Vectorstore is not initialized.'

        # Initialize an empty list to store the ids
        ids_for_target_file = []

        # Loop over the metadata
        for i in range(len(metadata['metadatas'])):
            # Check if the source matches the target file
            # We only compare the last part of the path (the actual file name)
            if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == files_samename:
                # If it matches, add the corresponding id to the list
                ids_for_target_file.append(metadata['ids'][i])

        # print("IDs for target file:", ids_for_target_file)
        try:
            self.vectorstore.delete(ids=ids_for_target_file)
        except ValueError as v:
            raise 'File does not exist in vectorstore.'
        return
    
    @property
    def load_file_names(self):
        try:
            vct_store = self.vectorstore.get()
            unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

            # Merge duplicate sources
            merged_sources = ', '.join(unique_sources)

            # Extract actual file names
            file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

            print(i18n("Successfully load kowledge base."))
            return file_names
        except IndexError:
            print(i18n('No file in vectorstore.'))
        return []