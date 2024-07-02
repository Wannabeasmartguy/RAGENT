import streamlit as st
import os
import whisper
from streamlit_float import *
from audiorecorder import audiorecorder

from configs.basic_config import I18nAuto, SUPPORTED_LANGUAGES


# TODO:后续使用 st.selectbox 替换,选项为 "English", "简体中文"
i18n = I18nAuto(language=SUPPORTED_LANGUAGES["简体中文"])


def float_chat_input_with_audio_recorder() -> str:
    """
    Create a container with a floating chat input and an audio recorder.
    """
    chat_input_container = st.container()
    with chat_input_container:
        character_input_column, voice_input_column = st.columns([0.9,0.1])
        character_input_placeholder = character_input_column.empty()
        prompt = character_input_placeholder.chat_input("What is up?")
        voice_input_popover = voice_input_column.popover(
            label="🎤"
        )
        voice_input_model_name = voice_input_popover.selectbox(
            label=i18n("Voice input model"),
            options=whisper.available_models(),
            index=3,
            key="voice_input_model"   
        )
        audio_recorder_container =  voice_input_popover.container(border=True)
        with audio_recorder_container:
            # TODO:没有麦克风可能无法录音
            # audio_recorded = audiorecorder(start_prompt='',stop_prompt='',pause_prompt='')
            audio_recorded = audiorecorder(pause_prompt='pause')
            audio_placeholder = st.empty()
            transcribe_button_placeholder = st.empty()
            if len(audio_recorded) > 0:
                # To play audio in frontend:
                audio = audio_recorded.export().read()
                audio_placeholder.audio(audio)
                transcribe_button = transcribe_button_placeholder.button(
                    label=i18n("Transcribe"),
                    use_container_width=True
                )
                # 临时存储音频文件
                with open("dynamic_configs/temp.wav", "wb") as f:
                    f.write(audio)
                # TODO：按下识别按钮后，才能识别语音
                # 加载语音识别模型
                if transcribe_button:
                    with st.status(i18n("Transcribing...")):
                        st.write(i18n("Loading model"))
                        voice_input_model = whisper.load_model(
                            name=voice_input_model_name,
                            download_root="./tts_models"
                        )
                        st.write(i18n("Model loaded"))
                        # 识别语音
                        st.write(i18n("Transcribing"))
                        transcribe_result = voice_input_model.transcribe(audio="dynamic_configs/temp.wav",word_timestamps=True,verbose=True)
                        st.write(i18n("Transcribed"))
                    st.write(transcribe_result.get("text","No result."))
                    # 删除临时文件
                    os.remove("dynamic_configs/temp.wav")

    chat_input_css = float_css_helper(bottom="6rem", display="flex", justify_content="center", margin="0 auto")
    chat_input_container.float(chat_input_css)
    return prompt