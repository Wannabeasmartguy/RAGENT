import os
import json
import logging
import locale
import streamlit as st

from typing import Literal


class I18nAuto:
    def __init__(self, i18n_dir: str, **kwargs):
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
        elif "language" in kwargs:
            config = {"language": kwargs["language"]}
        else:
            config = {}
        language = config.get("language", "auto")
        language = language.replace("-", "_")
        if language == "auto":
            language = locale.getdefaultlocale()[0] # get the language code of the system (ex. zh_CN)
        self.language_map = {}
        self.file_is_exists = os.path.isfile(os.path.join(i18n_dir, f"{language}.json"))
        if self.file_is_exists:
            with open(os.path.join(i18n_dir, f"{language}.json"), "r", encoding="utf-8") as f:
                self.language_map.update(json.load(f))
        else:
            logging.warning(
                f"Language file for {language} does not exist. Using English instead."
            )
            logging.warning(
                f"Available languages: {', '.join([x[:-5] for x in os.listdir(i18n_dir)])}"
            )
            with open(os.path.join(i18n_dir, "en_US.json"), "r", encoding="utf-8") as f:
                self.language_map.update(json.load(f))

    def __call__(self, key):
        if self.file_is_exists and key in self.language_map:
            return self.language_map[key]
        else:
            return key

