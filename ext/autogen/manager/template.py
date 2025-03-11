import os
import json
from typing import TypeVar

from ext.autogen.models.agent import AgentTemplate, ReflectionAgentTeamTemplate
from ext.autogen.config.constant.paths import AGENT_CONFIGS_DIR, AGENT_TEMPLATE_CONFIGS_FILE


class AgentTemplateFileManager:
    def __init__(self, user_id: str, template_file_path: str = AGENT_TEMPLATE_CONFIGS_FILE) -> None:
        self.template_file_path = template_file_path
        self.user_id = user_id
        self._check_template_file_exist_or_create()

    def _check_template_file_exist_or_create(self) -> None:
        os.makedirs(AGENT_CONFIGS_DIR, exist_ok=True)
        if not os.path.exists(self.template_file_path):
            with open(self.template_file_path, "w") as f:
                json.dump({}, f, indent=4)
    
    def _add_or_update_template_in_file(self, agent_template: AgentTemplate) -> None:
        with open(self.template_file_path, "r") as f:
            original_agent_template_config = json.load(f)
            original_agent_template_config[agent_template.id] = agent_template.to_dict()
            with open(self.template_file_path, "w") as f:
                json.dump(original_agent_template_config, f, indent=4)

    def create_agent_template(self, agent_template_config: dict) -> AgentTemplate:
        agent_template_config.update({"user_id": self.user_id})
        template_type = agent_template_config.get("template_type")
        if template_type == "reflection":
            agent_template = ReflectionAgentTeamTemplate.model_validate(agent_template_config)
        else:
            raise ValueError(f"Invalid template type: {template_type}")
        return agent_template

    def add_agent_template_to_file(self, agent_template: AgentTemplate) -> None:
        self._check_template_file_exist_or_create()
        self._add_or_update_template_in_file(agent_template)
    
    def update_agent_template_in_file(self, agent_template: AgentTemplate) -> None:
        self._check_template_file_exist_or_create()
        self._add_or_update_template_in_file(agent_template)

    def delete_agent_template_in_file(self, agent_template_id: str) -> None:
        with open(self.template_file_path, "r") as f:
            original_agent_template_config = json.load(f)
        if agent_template_id in original_agent_template_config:
            del original_agent_template_config[agent_template_id]
            with open(self.template_file_path, "w") as f:
                json.dump(original_agent_template_config, f, indent=4)
    
    @property
    def agent_templates(self) -> dict:
        with open(self.template_file_path, "r") as f:
            all_templates = json.load(f)
        # 根据user_id过滤
        return {k: v for k, v in all_templates.items() if v.get("user_id") == self.user_id}
