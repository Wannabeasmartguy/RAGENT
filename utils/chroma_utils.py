import chromadb
import tabulate
import os
import pandas as pd

from typing import Literal
from huggingface_hub import snapshot_download
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import chroma

from configs.basic_config import I18nAuto

i18n = I18nAuto()

def load_vectorstore(persist_vec_path:str,
                     embedding_model_type:Literal['OpenAI','Hugging Face(local)'],
                     embedding_model:str):
    '''
    Load vectorstore, and trun the files' name to dataframe.
    '''
    embedding_model_path = 'embedding model/'+embedding_model

    if persist_vec_path:
        if embedding_model_type == 'OpenAI':
            vectorstore = chroma.Chroma(persist_directory=persist_vec_path, 
                                embedding_function=AzureOpenAIEmbeddings(
                                                openai_api_type=os.getenv('API_TYPE'),
                                                azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                                                openai_api_key=os.getenv('AZURE_OAI_KEY'),
                                                openai_api_version=os.getenv('API_VERSION'),
                                                azure_deployment="text-embedding-ada-002",
                                        ))
        elif embedding_model_type == 'Hugging Face(local)':
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_path)
                vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                            embedding_function=embeddings)
            except:
                if embedding_model[:3] == 'bge':
                    snapshot_download(repo_id="BAAI/"+embedding_model,
                                      local_dir=embedding_model_path)
                    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_path)
                    vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                                embedding_function=embeddings)

    else:
        raise i18n("You didn't provide an absolute path to the knowledge base")

    try:
        vct_store = vectorstore.get()
        unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

        # Merge duplicate sources
        merged_sources = ', '.join(unique_sources)

        # Extract actual file names
        file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

        df = pd.DataFrame(file_names, columns=['文件名称'])
        print(i18n("Successfully load kowledge base."))
        return file_names
    except IndexError:
        print(i18n('No file in vectorstore.'))
        return []

def combine_lists_to_dicts(docs, ids, metas):
    """
    将三个列表的对应元素组合成一个个字典，然后将这些字典保存在一个列表中。

    参数:
    docs (list of str): 文档名列表
    ids (list of str): id列表
    metas (list of str): 元数据列表

    返回:
    list of dict: 每个字典包含三个键值对，键分别是"documents", "ids", "metadatas"，值来自对应的列表

    示例:
    combine_lists_to_dicts(["你好","hello"], ["sabea-12","asdao-141"], ["CoT.txt","abs.txt"])
    返回 [{"documents":"你好","ids":"sabea-12","metadatas":"CoT.txt"},{"documents":"hello","ids":"asdao-141","metadatas":"abs.txt"}]
    """

    # 使用zip函数将三个列表的对应元素打包成一个个元组
    tuples = zip(docs, ids, metas)

    # 将每个元组转换为字典，然后将这些字典保存在一个列表中
    dict_lists = [{"documents": doc, "ids": id, "metadatas": meta} for doc, id, meta in tuples]

    return dict_lists


def text_to_html(x, api=False):
    '''
    Encodes metadata in Chroma into HTML text for display in Gradio.
    
    Args:
        x: Metadata to be converted from ChromaDB
        api: Flag indicating if the function is called from an API
        
    Returns:
        str: HTML representation of the metadata
    '''
    x += "\n\n"
    if api:
        return x
    return """
            <style>
                .card-container {
                        display: flex;
                        flex-wrap: wrap;
                    }
                .card {
                    overflow-x: auto;
                    overflow-y: auto;
                    font-family: "微软雅黑", sans-serif;
                    font-size: 16px;
                    height: 200px;
                    width: 650px; 
                    white-space: pre-wrap;
                    white-space: -moz-pre-wrap;
                    white-space: -pre-wrap;
                    white-space: -o-pre-wrap;
                    word-wrap: break-word;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 10px;
                    margin: 10px;
                    background-color: #f9f9f9;
                    transition: box-shadow 0.3s;
                    box-sizing: border-box; /* 确保宽度包括边框和内边距 */
                }
                </style>
                <div class="card">
                %s
                </div>
""" % x


def dict_to_html(x:list[dict],file_name:str,advance_info:bool, small=True, api=False):
    info_list = []
    for info in x:
        if file_name in info['metadatas']['source']:
            df = pd.DataFrame(info.items(), columns=['Key', 'Value'])
            df.index = df.index + 1
            df.index.name = 'index'
            if api:
                res = tabulate.tabulate(df, headers='keys')
                doc_content = text_to_html(df.loc[1]['Value'])
                if advance_info:
                    info_list.append(res)
                info_list.append(doc_content)
            else:
                res = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
                doc_content = text_to_html(df.loc[1]['Value'])
                if advance_info:
                    info_list.append(res)
                info_list.append(doc_content)
    
    final_res = '\n\n'.join(info_list)

    # 用字符串的css,把最终的 html 高度限制在一页内，通过上下滚动查看完整内容
    if small:
        return f"""
            <style>
                .small-container {{
                    height: 650px;
                    overflow-x: auto;
                    overflow-y: auto;
                }}
            </style>
            <div class="small-container">
                <small>
                    {final_res}
                </small>
            </div>

            """
    else:
        return final_res
    

def get_chroma_file_info(persist_path:str,
                    file_name:str,
                    advance_info:bool,
                    collection_name:str="langchain",
                    limit:int=10000000000):
    '''
    Get the metadata of the file from the ChromaDB.
    
    Args:
        persist_path (str): Path to the ChromaDB database
        file_name (str): Name of the file to retrieve metadata for
        advance_info (bool): Flag indicating whether to include advanced information
        collection_name (str): Name of the collection in the ChromaDB
        limit (int): Maximum number of documents to retrieve
        
    Returns:
        str: HTML representation of the metadata
    '''
    try:
        client = chromadb.PersistentClient(path=persist_path)
        collection_lang = client.get_collection(collection_name)
    except ValueError:
        raise i18n("“Knowledge Base path” is empty, Please enter the path")
    metadata_pre10 = collection_lang.peek(limit=limit)  
    
    #get data for the first <limit> files 
    documents = metadata_pre10['documents']
    ids = metadata_pre10['ids']
    metadatas = metadata_pre10['metadatas']

    chroma_data_dic = combine_lists_to_dicts(documents, ids, metadatas)
    
    kb_info_html = dict_to_html(chroma_data_dic,file_name,advance_info)
    return kb_info_html

