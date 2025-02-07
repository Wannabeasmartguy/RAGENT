from .app import VERSION
from .i18n import SUPPORTED_LANGUAGES, I18N_DIR
from .paths import LOGO_DIR, KNOWLEDGE_BASE_DIR, EMBEDDING_OPTIONS_FILE_PATH
from .chat import (
    DEFAULT_DIALOG_TITLE,
    USER_AVATAR_SVG,
    AI_AVATAR_SVG,
)
from .prompts import (
    DEFAULT_SYSTEM_PROMPT, 
    ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
)
from .databases import (
    CHAT_HISTORY_DIR,
    CHAT_HISTORY_DB_FILE,
    CHAT_HISTORY_DB_TABLE,
    DYNAMIC_CONFIGS_DIR,
    OPENAI_LIKE_MODEL_CONFIG_FILE_PATH,
    EMBEDDING_DIR,
    EMBEDDING_CONFIG_FILE_PATH,
    RAG_CHAT_HISTORY_DB_TABLE,
    AGENT_CHAT_HISTORY_DB_TABLE
)

__all__ = [
    'VERSION',
    'SUPPORTED_LANGUAGES',
    'I18N_DIR',
    'LOGO_DIR',
    'KNOWLEDGE_BASE_DIR',
    'EMBEDDING_OPTIONS_FILE_PATH',
    'DEFAULT_DIALOG_TITLE',
    'DEFAULT_SYSTEM_PROMPT',
    'ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT',
    'CHAT_HISTORY_DIR',
    'CHAT_HISTORY_DB_FILE', 
    'CHAT_HISTORY_DB_TABLE',
    'DYNAMIC_CONFIGS_DIR',
    'OPENAI_LIKE_MODEL_CONFIG_FILE_PATH',
    'EMBEDDING_DIR',
    'EMBEDDING_CONFIG_FILE_PATH',
    'RAG_CHAT_HISTORY_DB_TABLE',
    'AGENT_CHAT_HISTORY_DB_TABLE',
    'SUMMARY_PROMPT',
    'USER_AVATAR_SVG',
    'AI_AVATAR_SVG',
]