DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

TOOL_USE_PROMPT = "You are an intelligent assistant that chooses whether or not to use a tool based on user commands."

ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT = """You are an intelligent assistant that chooses whether or not to use a tool based on user commands. 
If you use tools, just answer the question based on the output of the tool without any additional explanation. 
On the other hand, if you don't use tools, answer the question directly as best as you can.

Here are the other requirements you need to follow:

{user_system_prompt}"""

SUMMARY_PROMPT = """
You are an intelligent assistant that summarizes the conversation history.

Please summarize the conversation history in a concise manner.

**Rules:**

1. The summary should be concise and to the point without any additional explanation or punctuation mark.
2. The summary should be as short as possible.
3. The summary should be in user's language.

The conversation history is as follows:
"""
