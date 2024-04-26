import os
import json

from typing import Literal

from huggingface_hub import snapshot_download
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import chroma


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
                    snapshot_download(repo_id="BAAI/"+embedding_model,
                                      local_dir=local_embedding_model)
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