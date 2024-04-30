# RAGenT

[中文文档](README_zh.md) | **English**

Experience the power of Agent-powered models and Agent-driven knowledge bases in one click, without complex configuration.

## Features

Chat and Agent interactions:
- [x] 💭 Simple, easy-to-use chat box interface.
- [x] 💭 🌏️ Language options (Simplified Chinese, English)
- [x] 🔧 Inference support for multiple (local) model sources (Azure OpenAI, Groq, ollama, llamafile)
- [x] 🤖 Multiple Agent modes on-premises

Knowledgebase:
- [x] Optional embedding models (Hugging Face/OpenAI)
- [x] Easy-to-use knowledge base management
- [ ] Hybrid search, reranking, and specified file retrieval
- [ ] 📚️Agent-driven knowledge base

> If you like this project, please star it, it's the biggest encouragement for me!

## More details

General chat page:

![common chat in RAGenT](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/11f61cad-81eb-48f3-8e03-bab5a4bc9470)

Agent Conversation:

![agentchat-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e2cf62b3-447e-4b85-96dd-0bdd1e9e822f)

You can check out the specifics of the reflection:

![agentchat_expand_thought-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/ed33578e-e463-4eb6-996a-786c0d517eb3)

## Quick start

### Git

0. Use `git clone https://github.com/Wannabeasmartguy/RAGenT.git` to pull the code;
Then open your runtime environment in **command prompt (CMD)** and use `pip install -r requirements.txt` to install the runtime dependencies.

1. Configure the model dependencies: Modify the `.env_sample` file to `.env` and fill in the following:

    `AZURE_OAI_KEY` : If you are using an Azure OpenAI model, fill in the api key here;
    `AZURE_OAI_ENDPOINT` : If you are using an OpenAI model, fill in the end_point here;
    `GROQ_API_KEY` : if you are using Groq as a model source, fill in the api key here;

> If you are using Llamafile, please set the endpoint within the application after starting the Llamafile model.

2. launch the application:

Run: `streamlit run RAGenT.py` on the command line to start it.

## Route

- [ ] Increase the number of preset Agents
- [ ] Mixed retrieval, reordering and specified file retrieval
- [ ] 📚️Agent-driven Knowledge Base