import os

# 基础目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATABASE_DIR = os.path.join(ROOT_DIR, "databases")
KNOWLEDGE_BASE_DIR = os.path.join(DATABASE_DIR, "knowledgebase")
# TODO: 这两个暂时没用，可能会在后续版本中删除
CONFIGS_BASE_DIR = os.path.join(DATABASE_DIR, "configs")
CONFIGS_DB_FILE = os.path.join(CONFIGS_BASE_DIR, "configs.db")

# 资源目录
ASSETS_DIR = os.path.join(ROOT_DIR, "assets")

# 资源路径
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
STYLES_DIR = os.path.join(ASSETS_DIR, "styles")
LOCALE_DIR = os.path.join(ASSETS_DIR, "locale")
LOG_DIR = os.path.join(ROOT_DIR, "log")

# 各资源类型
LOGO_DIR = os.path.join(IMAGES_DIR, "logos")

# 配置文件路径
MODEL_CONFIG_DIR = os.path.join(CONFIG_DIR, "models")
EMBEDDING_OPTIONS_FILE_PATH = os.path.join(MODEL_CONFIG_DIR, "embedding_options.json")


if __name__ == "__main__":
    print(ROOT_DIR)
