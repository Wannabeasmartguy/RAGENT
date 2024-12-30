from abc import ABC, abstractmethod
from typing import Any, Dict, List

from autogen_agentchat.state import BaseState, TeamState


class BaseParser(ABC):
    @abstractmethod
    def parse_state_to_dict(self, state: TeamState) -> Dict[str, Any]:
        pass

    @abstractmethod
    def parse_dict_to_state(self, state_dict: Dict[str, Any]) -> BaseState:
        pass


class TeamStateParser(BaseParser):
    def parse_state_to_dict(self, state: TeamState) -> Dict[str, Any]:
        return state.model_dump()

    def parse_dict_to_state(self, state_dict: Dict[str, Any]) -> TeamState:
        return TeamState.model_validate(state_dict)

    def _parse_message_thread_to_openai_format(
        self, message_thread: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse the message thread to the format of OpenAI chat completion.
        """
        openai_format_messages = []
        for message in message_thread:
            openai_format_messages.append(
                {"role": message["source"], "content": message["content"]}
            )
        return openai_format_messages

    def parse_state_and_extract_history(
        self, state: TeamState, extract_agent_name: str = "group_chat_manager"
    ) -> List[str]:
        """
        Extract the history of the specified agent from the state.
        The history is a list of messages, each message is a dict with "source", "models_usage", "content" and "type" fields which are defined in the Autogen Message class.

        Args:
            state (TeamState): The state to extract the history from.
            extract_agent_name (str): The name of the agent to extract the history from. The default value "group_chat_manager" means the history of the team, because the team is created by the group chat manager.
            extract_agent_name can also be the name of a specific agent, it depends on the agent name when the agent is created.

        Returns:
            List[str]: The history of the specified agent.
        """
        agent_states = state.agent_states

        if extract_agent_name not in agent_states:
            raise ValueError(
                f"Agent name {extract_agent_name} not found in agent states"
            )
        
        for agent_id in agent_states.values():
            if extract_agent_name in agent_id:
                agent_state = agent_states[agent_id]
                break

        message_thread = agent_state.get("message_thread")
        if message_thread is None:
            raise ValueError(f"Message thread not found in agent state {agent_id}")
        chat_history = self._parse_message_thread_to_openai_format(message_thread)

        return chat_history
