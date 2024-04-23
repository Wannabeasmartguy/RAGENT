import os

from typing import List,Union
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from autogen.agentchat.contrib.capabilities import transform_messages, transforms


def reflection_agent_with_nested_chat(
        config_list: List[dict],
        max_message:int,
        work_dir:Union[str,Path,None]="workdir/coding",
        **kwargs
    ):
    '''
    Create a reflection agent with nested chat.Contain a user proxy agent, a writing assistant agent and a reflection_assistant.

    Args:
        config_list (List[dict]): List of configuration dictionaries for the LLM.
        max_message (int): Maximum number of messages will be sent in the chat history.
        work_dir (Union[str,Path,None]): Directory to store the generated code. Defaults to "workdir/coding".
        **kwargs: Additional keyword arguments for the reflection_assistant.
            max_tokens (int): Maximum number of tokens in the total chat history sent to the LLM. Defaults to None.
            max_tokens_per_message (int): Maximum number of tokens in each message sent to the LLM. Defaults to None.
    '''
    max_tokens = kwargs.get("max_tokens", None)
    max_tokens_per_message = kwargs.get("max_tokens_per_message", None)
    context_handling = transform_messages.TransformMessages(
        transforms=[
            transforms.MessageHistoryLimiter(max_messages=max_message),
            transforms.MessageTokenLimiter(max_tokens=max_tokens, max_tokens_per_message=max_tokens_per_message),
        ]
    )

    os.makedirs(work_dir, exist_ok=True)
    # Use DockerCommandLineCodeExecutor if docker is available to run the generated code.
    # Using docker is safer than running the generated code directly.
    # code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    try:
        code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    except:
        code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=10,
        code_execution_config={"executor": code_executor},
    )

    writing_assistant = AssistantAgent(
        name="writing_assistant",
        system_message="You are an writing assistant tasked to write engaging blogpost. You try generate the best blogpost possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts.",
        llm_config={"config_list": config_list, "cache_seed": None},
    )

    reflection_assistant = AssistantAgent(
        name="reflection_assistant",
        system_message="Generate critique and recommendations on the writing. Provide detailed recommendations, including requests for length, depth, style, etc..",
        llm_config={"config_list": config_list, "cache_seed": None},
    )

    def reflection_message(recipient, messages, sender, config):
        print("Reflecting...")
        return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


    context_handling.add_to_agent(writing_assistant)
    context_handling.add_to_agent(reflection_assistant)


    nested_chat_queue = [
        {
            "recipient": reflection_assistant,
            "message": reflection_message,
            "max_turns": 1,
        },
    ]
    user_proxy.register_nested_chats(
        nested_chat_queue,
        trigger=writing_assistant,
        # position=4,
    )

    # Use Cache.disk to cache the generated responses.
    # This is useful when the same request to the LLM is made multiple times.
    with Cache.disk(cache_seed=42) as cache:
        result = user_proxy.initiate_chat(
            writing_assistant,
            message="Write an engaging blogpost on the recent updates in AI. "
            "The blogpost should be engaging and understandable for general audience. "
            "Should have more than 3 paragraphes but no longer than 1000 words.",
            max_turns=2,
            cache=cache,
        )
        return result