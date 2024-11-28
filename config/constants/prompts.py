DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."

TOOL_USE_PROMPT = "You are an intelligent assistant that chooses whether or not to use a tool based on user commands."

ANSWER_USER_WITH_TOOLS_SYSTEM_PROMPT = """You are an intelligent assistant that chooses whether or not to use a tool based on user commands. 
If you use tools, just answer the question based on the output of the tool without any additional explanation. 
On the other hand, if you don't use tools, answer the question directly as best as you can.

Here are the other requirements you need to follow:

{user_system_prompt}"""