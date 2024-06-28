import os
import json
import logging
import locale
import streamlit as st


SUPPORTED_LANGUAGES = {
    "English": "en-US",
    "简体中文": "zh-CN",
}

DATABASE_DIR = os.path.join("databases")

KNOWLEDGE_BASE_DIR = os.path.join(DATABASE_DIR, "knowledgebase")

CONFIGS_BASE_DIR = os.path.join(DATABASE_DIR, "configs")
CONFIGS_DB_FILE = os.path.join(CONFIGS_BASE_DIR,"configs.db")
EMBEDDING_CONFIGS_DB_TABLE = "embedding_configs"
LLM_CONFIGS_DB_TABLE = "llm_configs"


class I18nAuto:
    def __init__(self,**kwargs):
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding='utf-8') as f:
                config = json.load(f)
        elif "language" in kwargs:
            config = {"language": kwargs["language"]}
        else:
            config = {}
        language = config.get("language", "auto")
        language = os.environ.get("LANGUAGE", language)
        language = language.replace("-", "_")
        if language == "auto":
            language = locale.getdefaultlocale()[0] # get the language code of the system (ex. zh_CN)
        self.language_map = {}
        self.file_is_exists = os.path.isfile(f"./locale/{language}.json")
        if self.file_is_exists:
            with open(f"./locale/{language}.json", "r", encoding="utf-8") as f:
                self.language_map.update(json.load(f))
        else:
            logging.warning(f"Language file for {language} does not exist. Using English instead.")
            logging.warning(f"Available languages: {', '.join([x[:-5] for x in os.listdir('./locale')])}")
            with open(f"./locale/en_US.json", "r", encoding="utf-8") as f:
                self.language_map.update(json.load(f))

    def __call__(self, key):
        if self.file_is_exists and key in self.language_map:
            return self.language_map[key]
        else:
            return key
        
def set_pages_configs_in_common(
        title,
        version:str="0.0.1",
        page_icon_path=os.path.dirname(__file__),
        init_sidebar_state="expanded"
    ):
    st.set_page_config(
        page_title=title,
        page_icon=page_icon_path,
        initial_sidebar_state=init_sidebar_state,
        menu_items={
                'Get Help': 'https://github.com/Wannabeasmartguy/RAGenT',
                'Report a bug': "https://github.com/Wannabeasmartguy/RAGenT/issues",
                'About': f"""欢迎使用 RAGenT WebUI {version}！"""
            }    
    )