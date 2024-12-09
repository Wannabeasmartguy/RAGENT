from config.constants.paths import ROOT_DIR
import os

I18N_DIR = os.path.join(ROOT_DIR, "assets", "locale")

# 默认语言，在I18nAuto中已经设置，这里不需要再设置
# DEFAULT_LANGUAGE = "en_US"
SUPPORTED_LANGUAGES = {
    "English": "en-US",
    "简体中文": "zh-CN",
}
