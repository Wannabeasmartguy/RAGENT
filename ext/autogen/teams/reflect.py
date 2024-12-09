from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models import OpenAIChatCompletionClient


class ReflectionTeamBuilder:
    def __init__(self):
        self.model_client = None
        self.primary_agent = None 
        self.critic_agent = None
        self.max_messages = 5
        self.termination_text = "APPROVE"

    def set_model_client(self, model_client):
        self.model_client = model_client
        return self

    def set_primary_agent(self, system_message="You are a helpful AI assistant."):
        self.primary_agent = AssistantAgent(
            "primary",
            model_client=self.model_client,
            system_message=system_message
        )
        return self

    def set_critic_agent(self, system_message="Provide constructive feedback. Respond with 'APPROVE' when your feedbacks are addressed."):
        self.critic_agent = AssistantAgent(
            "critic", 
            model_client=self.model_client,
            system_message=system_message
        )
        return self

    def set_max_messages(self, max_messages):
        self.max_messages = max_messages
        return self

    def set_termination_text(self, text):
        self.termination_text = text
        return self

    def build(self):
        if not all([self.model_client, self.primary_agent, self.critic_agent]):
            raise ValueError("Missing required components")

        text_termination = TextMentionTermination(self.termination_text)
        max_message_termination = MaxMessageTermination(self.max_messages)
        termination = text_termination | max_message_termination

        return RoundRobinGroupChat(
            [self.primary_agent, self.critic_agent], 
            termination_condition=termination
        )