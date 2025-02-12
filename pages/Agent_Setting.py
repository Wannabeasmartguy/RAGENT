import os
import json
import uuid
import asyncio

from config.constants import (
    LOGO_DIR, 
    I18N_DIR, 
    SUPPORTED_LANGUAGES
)
from core.basic_config import I18nAuto
from core.llm._client_info import (
    generate_client_config, 
    get_client_config_model,
    OpenAISupportedClients
)
from core.processors.config.llm import OAILikeConfigProcessor
from utils.basic_utils import (
    config_list_postprocess, 
    model_selector, 
    oai_model_config_selector
)
from utils.log.logger_config import setup_logger
from ext.autogen.models.agent import ReflectionAgentTeamTemplate, AgentTemplateType
from ext.autogen.manager.template import AgentTemplateFileManager

import streamlit as st
from loguru import logger


agent_template_file_manager = AgentTemplateFileManager()


def update_llm_config_list(key_suffix: str = ""):
    """更新llm配置列表，支持动态key
    
    Args:
        key_suffix (str): key的后缀，用于区分不同的配置
    """
    llm_config = generate_client_config(
        source=st.session_state[f"model_type{key_suffix}"].lower(),
        model=st.session_state[f"model{key_suffix}"],
        max_tokens=st.session_state[f"max_tokens{key_suffix}"],
        temperature=st.session_state[f"temperature{key_suffix}"],
        top_p=st.session_state[f"top_p{key_suffix}"],
        stream=st.session_state[f"if_stream{key_suffix}"],
    )
    if f"llm_config_list{key_suffix}" not in st.session_state:
        st.session_state[f"llm_config_list{key_suffix}"] = [llm_config.to_dict()]
    else:
        st.session_state[f"llm_config_list{key_suffix}"][0] = llm_config.to_dict()


def create_agent_team_form(
    form_key: str = "create_agent_team",
    template_id: str = None,
):
    agent_team_form = st.form(i18n("Create Agent Team") + f"_{form_key}")
    
    if template_id:
        template = agent_template_file_manager.agent_templates.get(template_id)
        if template:
            st.session_state[f"agent_team_type_{form_key}"] = template["team_type"].capitalize()
            st.session_state[f"agent_team_name_{form_key}"] = template["name"]
            st.session_state[f"agent_team_description_{form_key}"] = template["description"]
            
            # 根据团队类型加载特定字段
            if template["team_type"].lower() == AgentTemplateType.REFLECTION.value:
                st.session_state[f"primary_agent_system_message_{form_key}"] = template["primary_agent_system_message"]
                st.session_state[f"critic_agent_system_message_{form_key}"] = template["critic_agent_system_message"]
                st.session_state[f"max_messages_{form_key}"] = template["max_messages"]
                st.session_state[f"termination_text_{form_key}"] = template["termination_text"]
    
    team_type_column, team_name_column = agent_team_form.columns([0.5, 0.5])
    with team_type_column:
        agent_team_type = team_type_column.selectbox(
            i18n("Team Type"), 
            options=[team_type.value.capitalize() for team_type in AgentTemplateType],
            key=f"agent_team_type_{form_key}"
        )
    with team_name_column:
        agent_team_name = team_name_column.text_input(
            i18n("Team Name"), 
            key=f"agent_team_name_{form_key}"
        )
    agent_team_description = agent_team_form.text_input(
        i18n("Team Description"), 
        key=f"agent_team_description_{form_key}"
    )

    # 根据选择的团队类型显示不同的配置项
    selected_team_type = st.session_state[f"agent_team_type_{form_key}"].lower()
    
    if selected_team_type == AgentTemplateType.REFLECTION.value:
        primary_agent_system_message = agent_team_form.text_area(
            i18n("Primary Agent System Message"), 
            key=f"primary_agent_system_message_{form_key}"
        )
        critic_agent_system_message = agent_team_form.text_area(
            i18n("Critic Agent System Message"), 
            key=f"critic_agent_system_message_{form_key}"
        )
        max_messages = agent_team_form.number_input(
            i18n("Max Messages"), 
            min_value=1,
            value=10,
            key=f"max_messages_{form_key}"
        )
        termination_text = agent_team_form.text_input(
            i18n("Termination Text"), 
            key=f"termination_text_{form_key}"
        )
    # 未来可以添加其他类型的配置项
    # elif selected_team_type == AgentTemplateType.DEBATE.value:
    #     ...
    
    def submit_and_create_template_button_callback():
        # 基础字段验证
        if st.session_state[f"agent_team_name_{form_key}"] == "":
            st.toast(i18n("Please fill in team name"), icon="❌")
            return
            
        # 根据团队类型验证特定字段
        if selected_team_type == AgentTemplateType.REFLECTION.value:
            if (
                st.session_state[f"primary_agent_system_message_{form_key}"] == "" 
                or st.session_state[f"critic_agent_system_message_{form_key}"] == "" 
                or st.session_state[f"termination_text_{form_key}"] == ""
            ):
                st.toast(i18n("Please fill in all required fields for Reflection team"), icon="❌")
                return
        
        # 设置基础字段
        st.session_state.agent_team_name = st.session_state[f"agent_team_name_{form_key}"]
        st.session_state.agent_team_description = st.session_state[f"agent_team_description_{form_key}"]
        st.session_state.agent_team_type = st.session_state[f"agent_team_type_{form_key}"]
        
        # 根据团队类型设置特定字段
        template_config = {
            "id": template_id if template_id else str(uuid.uuid4()),
            "name": st.session_state.agent_team_name,
            "description": st.session_state.agent_team_description if st.session_state.agent_team_description != "" else i18n("No description"),
            "llm": get_client_config_model(st.session_state[f"llm_config_list_{form_key}"][0]),
            "template_type": st.session_state.agent_team_type.lower(),
        }
        logger.debug(f"template_config: {template_config}")
        
        if selected_team_type == AgentTemplateType.REFLECTION.value:
            template_config.update({
                "primary_agent_system_message": st.session_state[f"primary_agent_system_message_{form_key}"],
                "critic_agent_system_message": st.session_state[f"critic_agent_system_message_{form_key}"],
                "max_messages": st.session_state[f"max_messages_{form_key}"],
                "termination_text": st.session_state[f"termination_text_{form_key}"],
            })
        
        try:
            st.session_state.agent_team_template = agent_template_file_manager.create_agent_template(
                agent_template_config=template_config
            )
            agent_template_file_manager.add_agent_template_to_file(st.session_state.agent_team_template)
            st.toast(i18n("Agent team updated successfully") if template_id else i18n("Agent team created successfully"), icon="✅")
            
            if template_id:
                st.session_state[f"edit_mode_{template_id}"] = False
            
            # 清空表单通用字段
            st.session_state[f"agent_team_name_{form_key}"] = ""
            st.session_state[f"agent_team_description_{form_key}"] = ""
            # 清空特定字段
            if selected_team_type == AgentTemplateType.REFLECTION.value:
                st.session_state[f"primary_agent_system_message_{form_key}"] = ""
                st.session_state[f"critic_agent_system_message_{form_key}"] = ""
                st.session_state[f"max_messages_{form_key}"] = 10
                st.session_state[f"termination_text_{form_key}"] = ""
            
        except Exception as e:
            st.toast(i18n(f"Failed to {'update' if template_id else 'create'} agent team: {str(e)}"), icon="❌")
            logger.error(f"Failed to {'update' if template_id else 'create'} agent team: {e}")

    submit_and_create_template_button = agent_team_form.form_submit_button(
        i18n("Submit"),
        on_click=submit_and_create_template_button_callback,
    )


async def create_agent_template_card_gallery(
    columns: int = 3
):
    """
    创建一个包含所有模板配置的卡片画廊。
    
    :param columns: 每行显示的卡片数量
    """
    # 读取模板配置文件
    agent_template_config = agent_template_file_manager.agent_templates

    if not agent_template_config:
        st.info(i18n("No agent templates found. Please create one first."))
        return
        
    # 多于3个卡片时，每行显示3个卡片，少于3个卡片时，显示全部
    num_columns = min(columns, len(agent_template_config))
    rows = [st.columns(num_columns) for _ in range((len(agent_template_config) + 2) // num_columns)]
    
    # 遍历所有模板配置
    for (template_id, template), (row_idx, col) in zip(
        agent_template_config.items(),
        [(i,j) for i,row in enumerate(rows) for j in range(len(row))]
    ):
        with rows[row_idx][col].container(border=True):
            st.subheader(template["name"] if template["name"] else i18n("Unnamed Template"))
            
            if template["description"]:
                st.write(template["description"])
                
            with st.expander(i18n("Template Details")):
                st.write(f"**{i18n('Team Type')}:** {template['team_type']}")
                st.write(f"**{i18n('Model')}:** {template['llm']['model']}")
                st.write(f"**{i18n('Primary Agent System Message')}:** {template['primary_agent_system_message']}")
                st.write(f"**{i18n('Critic Agent System Message')}:** {template['critic_agent_system_message']}")
                st.write(f"**{i18n('Max Messages')}:** {template['max_messages']}")
                st.write(f"**{i18n('Termination Text')}:** {template['termination_text']}")
                
            col1, col2 = st.columns(2)
            with col2:
                if st.button(i18n("Delete"), key=f"delete_{template_id}", use_container_width=True):
                    agent_template_file_manager.delete_agent_template_in_file(template_id)
                    st.rerun()
            with col1:
                if st.button(i18n("Edit"), key=f"edit_{template_id}", use_container_width=True):
                    st.session_state[f"edit_mode_{template_id}"] = True
                    # 初始化编辑模式下的模型配置
                    if f"llm_config_list_edit_{template_id}" not in st.session_state:
                        st.session_state[f"llm_config_list_edit_{template_id}"] = [template["llm"]]
                        # 设置模型类型
                        st.session_state[f"model_type_edit_{template_id}"] = template["llm"]["config_type"]
                    st.rerun()

            # 编辑模式下显示编辑表单
            if st.session_state.get(f"edit_mode_{template_id}", False):
                # 关闭其他所有卡片
                for key in st.session_state.keys():
                    if key.startswith("edit_mode_") and key != f"edit_mode_{template_id}":
                        st.session_state[key] = False
                # 使用 create_model_select_container 创建模型选择器
                create_model_select_container(key_suffix=f"_edit_{template_id}")
                # 创建编辑表单
                create_agent_team_form(form_key=f"edit_{template_id}", template_id=template_id)
                close_button = st.button(i18n("Close Edit"), key=f"close_{template_id}", use_container_width=True)
                if close_button:
                    st.session_state[f"edit_mode_{template_id}"] = False
                    st.rerun()


language = os.getenv("LANGUAGE", "简体中文")
i18n = I18nAuto(
    i18n_dir=I18N_DIR,
    language=SUPPORTED_LANGUAGES[language]
)

def create_model_select_container(key_suffix: str = ""):
    """创建模型选择容器
    
    Args:
        key_suffix (str): key的后缀，用于避免重复key。例如在编辑模式下使用 '_edit_template_id' 作为后缀
    """
    model_select_container = st.container(border=True)
    with model_select_container:
        llm_config_container = model_select_container.container()

        model_type_column, model_placeholder_column = llm_config_container.columns([0.5, 0.5])
        with model_type_column:
            model_type = model_type_column.selectbox(
                label=i18n("Model type"),
                options=[provider.value for provider in OpenAISupportedClients],
                format_func=lambda x: x.capitalize(),
                on_change=lambda: update_llm_config_list(key_suffix),
                key=f"model_type{key_suffix}"
            )
        with model_placeholder_column:
            model_placeholder = model_placeholder_column.empty()
        with st.expander(label=i18n("Model config"), expanded=True):
            # 处理默认值
            default_max_tokens = 1900
            default_temperature = 0.5
            default_top_p = 0.5
            default_stream = True

            if f"llm_config_list{key_suffix}" in st.session_state:
                # 如果卡片的配置列表存在，说明已点击编辑键，则加载进来
                default_max_tokens = config_list_postprocess(st.session_state[f"llm_config_list{key_suffix}"])[0].get("max_tokens", 1900)
                default_temperature = config_list_postprocess(st.session_state[f"llm_config_list{key_suffix}"])[0].get("temperature", 0.5)
                default_top_p = config_list_postprocess(st.session_state[f"llm_config_list{key_suffix}"])[0].get("top_p", 0.5)
                default_stream = config_list_postprocess(st.session_state[f"llm_config_list{key_suffix}"])[0].get("stream", True)

            max_tokens = llm_config_container.number_input(
                label=i18n("Max tokens"),
                min_value=1,
                value=default_max_tokens,
                step=1,
                on_change=lambda: update_llm_config_list(key_suffix),
                key=f"max_tokens{key_suffix}",
                help=i18n(
                    "Maximum number of tokens to generate in the completion.Different models may have different constraints, e.g., the Qwen series of models require a range of [0,2000)."
                ),
            )
            temperature_column, top_p_column = llm_config_container.columns([0.5, 0.5])
            temperature = temperature_column.slider(
                label=i18n("Temperature"),
                min_value=0.0,
                max_value=1.0,
                value=default_temperature,
                step=0.1,
                key=f"temperature{key_suffix}",
                on_change=lambda: update_llm_config_list(key_suffix),
                help=i18n(
                    "'temperature' controls the randomness of the model. Lower values make the model more deterministic and conservative, while higher values make it more creative and diverse. The default value is 0.5."
                ),
            )
            top_p = top_p_column.slider(
                label=i18n("Top p"),
                min_value=0.0,
                max_value=1.0,
                value=default_top_p,
                step=0.1,
                key=f"top_p{key_suffix}",
                on_change=lambda: update_llm_config_list(key_suffix),
                help=i18n(
                    "Similar to 'temperature', but don't change it at the same time as temperature"
                ),
            )
            if_stream = llm_config_container.toggle(
                label=i18n("Stream"),
                value=default_stream,
                key=f"if_stream{key_suffix}",
                on_change=lambda: update_llm_config_list(key_suffix),
                help=i18n(
                    "Whether to stream the response as it is generated, or to wait until the entire response is generated before returning it. If it is disabled, the model will wait until the entire response is generated before returning it."
                ),
            )
        
        if model_type != OpenAISupportedClients.OPENAI_LIKE.value:
            def get_selected_non_llamafile_model_index(model_type) -> int:
                try:
                    model = st.session_state[f"llm_config_list{key_suffix}"][0].get("model")
                    logger.debug(f"model get: {model}")
                    if model:
                        options = model_selector(model_type)
                        if model in options:
                            options_index = options.index(model)
                            logger.debug(
                                f"model {model} in options, index: {options_index}"
                            )
                            return options_index
                        else:
                            st.session_state[f"llm_config_list{key_suffix}"][0].update(
                                {"model": options[0]}
                            )
                            logger.debug(
                                f"model {model} not in options, set model in config list to first option: {options[0]}"
                            )
                            return 0
                except (ValueError, AttributeError, IndexError):
                    logger.warning(
                        f"Model {model} not found in model_selector for {model_type}, returning 0"
                    )
                    return 0

            model = model_placeholder.selectbox(
                label=i18n("Model"),
                options=model_selector(st.session_state[f"model_type{key_suffix}"]),
                index=get_selected_non_llamafile_model_index(
                    st.session_state[f"model_type{key_suffix}"]
                ),
                key=f"model{key_suffix}",
                on_change=lambda: update_llm_config_list(key_suffix),
            )
        elif model_type == OpenAISupportedClients.OPENAI_LIKE.value:
            def get_selected_llamafile_model() -> str:
                if st.session_state[f"llm_config_list{key_suffix}"]:
                    return st.session_state[f"llm_config_list{key_suffix}"][0].get("model")
                else:
                    logger.warning("llm_config_list is empty, using default model")
                    return oai_model_config_selector(
                        st.session_state.oai_like_model_config_dict
                    )[0]

            model = model_placeholder.text_input(
                label=i18n("Model"),
                value=get_selected_llamafile_model(),
                key=f"model{key_suffix}",
                placeholder=i18n("Fill in custom model name. (Optional)"),
                on_change=lambda: update_llm_config_list(key_suffix),
            )
            with llm_config_container.popover(
                label=i18n("Llamafile config"), use_container_width=True
            ):

                def get_selected_llamafile_endpoint() -> str:
                    try:
                        return st.session_state[f"llm_config_list{key_suffix}"][0].get("base_url")
                    except:
                        return oai_model_config_selector(
                            st.session_state.oai_like_model_config_dict
                        )[1]

                llamafile_endpoint = st.text_input(
                    label=i18n("Llamafile endpoint"),
                    value=get_selected_llamafile_endpoint(),
                    key=f"llamafile_endpoint{key_suffix}",
                    type="password",
                    on_change=lambda: update_llm_config_list(key_suffix),
                )

                def get_selected_llamafile_api_key() -> str:
                    try:
                        return st.session_state[f"llm_config_list{key_suffix}"][0].get("api_key")
                    except:
                        return oai_model_config_selector(
                            st.session_state.oai_like_model_config_dict
                        )[2]

                llamafile_api_key = st.text_input(
                    label=i18n("Llamafile API key"),
                    value=get_selected_llamafile_api_key(),
                    key=f"llamafile_api_key{key_suffix}",
                    type="password",
                    placeholder=i18n("Fill in your API key. (Optional)"),
                    on_change=lambda: update_llm_config_list(key_suffix),
                )

                def save_oai_like_config_button_callback():
                    config_id = oailike_config_processor.update_config(
                        model=model,
                        base_url=llamafile_endpoint,
                        api_key=llamafile_api_key,
                        description=st.session_state.get("config_description", ""),
                    )
                    st.toast(i18n("Model config saved successfully"), icon="✅")
                    return config_id

                config_description = st.text_input(
                    label=i18n("Config Description"),
                    key=f"config_description{key_suffix}",
                    placeholder=i18n("Enter a description for this configuration"),
                    on_change=lambda: update_llm_config_list(key_suffix),
                )

                save_oai_like_config_button = st.button(
                    label=i18n("Save model config"),
                    on_click=save_oai_like_config_button_callback,
                    use_container_width=True,
                )

                st.write("---")

                config_list = oailike_config_processor.list_model_configs()
                config_options = [
                    f"{config['model']} - {config['description']}"
                    for config in config_list
                ]

                selected_config = st.selectbox(
                    label=i18n("Select model config"),
                    options=config_options,
                    format_func=lambda x: x,
                    on_change=lambda: st.toast(
                        i18n("Click the Load button to apply the configuration"),
                        icon="🚨",
                    ),
                    key=f"selected_config{key_suffix}",
                )

                def load_oai_like_config_button_callback():
                    selected_index = config_options.index(
                        selected_config
                    )
                    selected_config_id = config_list[selected_index]["id"]

                    logger.info(f"Loading model config: {selected_config_id}")
                    config = oailike_config_processor.get_model_config(
                        config_id=selected_config_id
                    )

                    if config:
                        config_data = config  # 不再需要 next(iter(config.values()))
                        st.session_state.oai_like_model_config_dict = {
                            config_data["model"]: config_data
                        }
                        st.session_state.model = config_data["model"]
                        st.session_state.llamafile_endpoint = config_data["base_url"]
                        st.session_state.llamafile_api_key = config_data["api_key"]
                        st.session_state.config_description = config_data.get(
                            "description", ""
                        )

                        logger.info(
                            f"Llamafile Model config loaded: {st.session_state.oai_like_model_config_dict}"
                        )

                        st.session_state[f"llm_config_list{key_suffix}"][0]["model"] = config_data["model"]
                        st.session_state[f"llm_config_list{key_suffix}"][0]["api_key"] = config_data["api_key"]
                        st.session_state[f"llm_config_list{key_suffix}"][0]["base_url"] = config_data["base_url"]

                        logger.info(
                            f"Agent team's llm config list updated: {st.session_state[f'llm_config_list{key_suffix}']}"
                        )
                        st.toast(i18n("Model config loaded successfully"), icon="✅")
                    else:
                        st.toast(i18n("Failed to load model config"), icon="❌")

                load_oai_like_config_button = st.button(
                    label=i18n("Load model config"),
                    use_container_width=True,
                    type="primary",
                    on_click=load_oai_like_config_button_callback,
                )

                def delete_oai_like_config_button_callback():
                    selected_index = config_options.index(
                        st.session_state.selected_config
                    )
                    selected_config_id = config_list[selected_index]["id"]
                    oailike_config_processor.delete_model_config(selected_config_id)
                    st.toast(i18n("Model config deleted successfully"), icon="🗑️")
                    # st.rerun()

                delete_oai_like_config_button = st.button(
                    label=i18n("Delete model config"),
                    use_container_width=True,
                    on_click=delete_oai_like_config_button_callback,
                )

        reset_model_button = llm_config_container.button(
            label=i18n("Reset model info"),
            on_click=lambda x: x.cache_clear(),
            args=(model_selector,),
            key=f"reset_model_button{key_suffix}",
            use_container_width=True,
        )
        load_model_button = llm_config_container.button(
            label=i18n("Load model setting"),
            on_click=lambda: update_llm_config_list(key_suffix),
            key=f"load_model_button{key_suffix}",
            use_container_width=True,
            type="primary",
        )
        if load_model_button:
            st.toast(i18n("Model setting loaded successfully"), icon="✅")


logo_path = os.path.join(LOGO_DIR, "RAGENT_logo.png")
logo_text = os.path.join(LOGO_DIR, "RAGENT_logo_with_text_horizon.png")

oailike_config_processor = OAILikeConfigProcessor()

st.set_page_config(
    page_title="Agent Setting",
    page_icon=logo_path,
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.logo(logo_text, icon_image=logo_path)

    st.page_link("RAGENT.py", label="💭 Chat")
    st.page_link("pages/RAG_Chat.py", label="🧩 RAG Chat")
    st.page_link("pages/1_🤖AgentChat.py", label="🤖 Agent Chat")

st.title(i18n("Agent Setting"))

agent_list_tab, create_agent_form_tab = st.tabs([i18n("Agent List"), i18n("Create Agent")])

with agent_list_tab:
    asyncio.run(create_agent_template_card_gallery(2))

with create_agent_form_tab:
    # 初始化 llm_config_list_new
    if "llm_config_list_new" not in st.session_state:
        llm_config = generate_client_config(
            source=OpenAISupportedClients.AOAI.value,
            model=model_selector(OpenAISupportedClients.AOAI.value)[0],
            stream=True,
        )
        st.session_state["llm_config_list_new"] = [llm_config.to_dict()]
    
    create_model_select_container(key_suffix="_new")
    
    st.write("## " + i18n("Create Agent Team"))
    with st.container(border=True):
        create_agent_team_form(form_key="new")
