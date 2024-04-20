from autogen.agentchat import AssistantAgent, UserProxyAgent,ConversableAgent
from autogen.coding import DockerCommandLineCodeExecutor

import os
from dotenv import load_dotenv
from pathlib import Path

from aoai.tools.tools import *

load_dotenv()


def create_two_agent_chat(query: str,
                          config_list: list[dict]):
    """
    Create the LLM config.
    
    Args:
        config_list (list[dict]): The list of configs.
        
    Returns:

    """
    llm_config = {
        "config_list": config_list,
    }

    # Let's first define the assistant agent that suggests tool calls.
    assistant = ConversableAgent(
        name="Assistant",
        system_message="You are a helpful AI assistant. "
        "You can help with simple calculations. "
        "Return 'TERMINATE' when the task is done.",
        llm_config=llm_config,
    )

    # The user proxy agent is used for interacting with the assistant agent
    # and executes tool calls.
    user_proxy = ConversableAgent(
        name="User",
        llm_config=False,
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
    )

    assistant.register_for_llm(name='calculator',description="A simple calculator")(calculator)
    user_proxy.register_for_execution(name='calculator')(calculator)

    from autogen.agentchat import register_function

    # Register the calculator function to the two agents.
    register_function(
        web_scraper,
        caller=assistant,  # The assistant agent can suggest calls to the calculator.
        executor=user_proxy,  # The user proxy agent can execute the calculator calls.
        name="web_scraper",  # By default, the function name is used as the tool name.
        description="Useful to scrape web pages, and extract text content.",  # A description of the tool.
    )


    # Let's create the agent chat.
    result = user_proxy.initiate_chat(assistant,message=query)

    return result