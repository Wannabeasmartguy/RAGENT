# RAGenT

**English** | [中文文档](README_zh.md)

Experience the power of Agent-powered models and Agent-driven knowledge bases in one click, without complex configuration.

![image](https://github.com/Wannabeasmartguy/RAGenT/assets/107250451/26bb3f1e-e784-4e48-9b09-7050c3e98d27)

## Features

Chat and Agent interactions:
- [x] 💭 Simple, easy-to-use chat box interface.
- [x] 🌏️ Language options (Simplified Chinese, English)
- [x] 🔧 Inference support for multiple (local) model sources (Azure OpenAI, Groq, ollama, llamafile)
- [x] 🤖 Multiple Agent modes on-premises
- [x] 🖥️ Local storage of dialog data and management

Knowledgebase:
- [x] Optional embedding models (Hugging Face/OpenAI)
- [x] Easy-to-use knowledge base management
- [x] Hybrid search, reranking, and specified file retrieval

> If you like this project, please star it, it's the biggest encouragement for me!

## More details

Agent Conversation:

Function Call:
![function call culculator](https://github.com/Wannabeasmartguy/RAGenT/assets/107250451/e457586a-f52d-41b2-950b-77bb08c42c94)

Reflection:
![agentchat-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e2cf62b3-447e-4b85-96dd-0bdd1e9e822f)

You can check out the specifics of the reflection:

![agentchat_expand_thought-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/ed33578e-e463-4eb6-996a-786c0d517eb3)

## Quick start

### Git

0. Use `git clone https://github.com/Wannabeasmartguy/RAGenT.git` to pull the code;
Then open your runtime environment in **command prompt (CMD)** and use `pip install -r requirements.txt` to install the runtime dependencies.

1. Configure the model dependencies: Modify the `.env_sample` file to `.env` and fill in the following:

    - `LANGUAGE`: Support `English` and `简体中文`, defualt is `English`;
    - `AZURE_OAI_KEY` : If you are using an Azure OpenAI model, fill in the api key here;
    - `AZURE_OAI_ENDPOINT` : If you are using an OpenAI model, fill in the end_point here;
    - `API_VERSION`: If you are using an Azure OpenAI model, fill in the api version here;
    - `API_TYPE`: if you are using an Azure OpenAI model, fill in the api type here;
    - `GROQ_API_KEY` : if you are using Groq as the model source, fill in the api key here;
    - `COZE_ACCESS_TOKEN`: if you need to use the created Coze Bot, fill in the access token here;

> If you are using Llamafile, please set the endpoint within the application after starting the Llamafile model.

2. launch the application:

Run: Run `python startup.py` on the command line can start it.

## Route

- [ ] Chat history and configuration local persistence
    - [x] Chat history local persistence
    - [ ] Configuration local persistence
- [ ] Increase the number of preset Agents
- [ ] Mixed retrieval, reordering and specified file retrieval
- [ ] 📚️Agent-driven Knowledge Base
