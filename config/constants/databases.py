import os
from config.constants.paths import ROOT_DIR

# 数据库目录
DATABASE_DIR = os.path.join(ROOT_DIR, "databases")

# 知识库及嵌入模型配置信息
EMBEDDING_CONFIG_FILE_PATH = os.path.join(
    ROOT_DIR, "dynamic_configs", "embedding_config.json"
)

# 聊天记录目录
CHAT_HISTORY_DIR = os.path.join(DATABASE_DIR, "chat_history")
# 知识库目录
KNOWLEDGE_BASE_DIR = os.path.join(DATABASE_DIR, "knowledgebase")
# 嵌入模型目录
EMBEDDING_DIR = os.path.join(DATABASE_DIR, "embeddings")

# 配置目录
CONFIGS_BASE_DIR = os.path.join(DATABASE_DIR, "configs")
CONFIGS_DB_FILE = os.path.join(CONFIGS_BASE_DIR, "configs.db")

# 聊天记录数据库文件
CHAT_HISTORY_DB_FILE = os.path.join(CHAT_HISTORY_DIR, "chat_history.db")
# 聊天记录表名称
CHAT_HISTORY_DB_TABLE = "chatbot_chat_history"
# RAG聊天记录表名称
RAG_CHAT_HISTORY_DB_TABLE = "custom_rag_chat_history"

# 配置表名称
EMBEDDING_CONFIGS_DB_TABLE = "embedding_configs"
LLM_CONFIGS_DB_TABLE = "llm_configs"
