# RAGenT

**English** | [中文文档](./docs/README_zh.md) | [日本語](./docs/README_ja.md)

Probably one of the lightest native RAG + Agent apps out there，experience the power of Agent-powered models and Agent-driven knowledge bases in one click, without complex configuration.

![image](https://github.com/user-attachments/assets/f50c9b86-55c8-4881-a7cf-b2ccf3b35ece)

## Features

Chat and Agent interactions:
- [x] 💭 Simple, easy-to-use chat box interface.
- [x] 🌏️ Language options (Simplified Chinese, English)
- [x] 🔧 Inference support for multiple (local) model sources (Azure OpenAI, Groq, ollama, llamafile)
- [x] Native Function Call (OpenAI, Azure OpenAI, OpenAI Like, Ollama)
- [x] 🤖 Multiple Agent modes on-premises
- [x] 🖥️ Local storage of dialog data and management
  - [x] Multiple export formats(Markdown, HTML)
  - [x] Multiple themes(HTML)

Knowledgebase:
- [x] **Native implementation** of Retrieval Augmentation Generation (RAG), lightweight and efficient
- [x] Optional embedding models (Hugging Face/OpenAI)
- [x] Easy-to-use knowledge base management
- [x] Hybrid search, reranking, and specified file retrieval

> If you like this project, please star it, it's the biggest encouragement for me!

## More details

### General

#### Voice to text input:

<img src="https://github.com/user-attachments/assets/37ea413d-5ef6-4783-a2da-ed6d1d010f58" alt="image" style="zoom:50%;" />

#### Export

Support export format, theme selection and export range control:

<img src="https://github.com/user-attachments/assets/f4e1461e-2334-4b45-b4d9-2ff0d79a0e63" alt="Export settings and preview" style="zoom:40%;" />

Currently supported themes: 

| Default | Glassmorphism |
| :-----: | :-----------: |
| <img src="https://github.com/user-attachments/assets/6ac8132c-0821-4487-9a1a-a0297a35783a" alt="default theme" width="300" /> | <img src="https://github.com/user-attachments/assets/87b07e86-dd98-4e66-a850-17b776fbeb1c" alt="Glassmorphism theme" width="300" /> |



### RAG Chat

Set up the model (sidebar) and view detailed references:

<img src="https://github.com/user-attachments/assets/4fba2259-3362-42b2-a4d5-85e0658d7720" alt="image" style="zoom:50%;" />

Configure RAG ：

<img src="https://github.com/user-attachments/assets/565d96dc-3f42-4f7d-a368-55af9f4a5d77" alt="image" style="zoom:50%;" />

### Function Call

Function calls are supported on both `Chat` and `AgentChat` pages, but are implemented differently.

#### Chat Page

The Function Calls on this page are native and work for all OpenAI Compatible models, but require the model itself to support Function calls.

<img src="https://github.com/user-attachments/assets/75163c4d-bcd2-4ef0-83d5-ab27c6527715" alt="image" style="zoom:50%;" />

> You can also customize the function you want to call, please refer to [toolkits.py](tools/toolkits.py) for writing rules.

#### AgentChat Page

Relying on the AutoGen framework for implementation (testing), please refer to the documentation of [AutoGen](https://github.com/microsoft/autogen) for model compatibility.

Function call can significantly enhance the capabilities of LLM and currently supports OpenAI, Azure OpenAI, Groq, and local models.（[by LiteLLM + Ollama](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama#using-litellmollama-with-autogen)）。

<img src="https://github.com/user-attachments/assets/4eabcedb-5717-46b1-b2f4-4324b5f1fb67" alt="openai function call" style="zoom:50%;" />

> You can also customize the function you want to call, please note that AutoGen's function writing is **different** from the native function calling writing rules, please refer to the [Official Documentation](https://microsoft.github.io/autogen/docs/tutorial/tool-use/) and this project's [tools.py](llm/aoai/tools/tools.py).

## Quick start

### Git

0. Use `git clone https://github.com/Wannabeasmartguy/RAGenT.git` to pull the code;
Then open your runtime environment in **command prompt (CMD)** and use `pip install -r requirements.txt` to install the runtime dependencies.

1. Configure the model dependencies: Modify the `.env_sample` file to `.env` and fill in the following:

    - `LANGUAGE`: Support `English` and `简体中文`, defualt is `English`;
    - `OPENAI_API_KEY` : If you are using an OpenAI model, fill in the api key here;
    - `AZURE_OAI_KEY` : If you are using an Azure OpenAI model, fill in the api key here;
    - `AZURE_OAI_ENDPOINT` : If you are using an OpenAI model, fill in the end_point here;
    - `API_VERSION`: If you are using an Azure OpenAI model, fill in the api version here;
    - `API_TYPE`: if you are using an Azure OpenAI model, fill in the api type here;
    - `GROQ_API_KEY` : if you are using Groq as the model source, fill in the api key here;
    - `COZE_ACCESS_TOKEN`: if you need to use the created Coze Bot, fill in the access token here;

> If you are using Llamafile, please set the endpoint within the application after starting the Llamafile model.

2. launch the application:

Run: Run `streamlit run RAGenT.py` on the command line can start it.

## Route

- [x] Chat history and configuration local persistence
    - [x] Chat history local persistence
    - [x] Configuration local persistence
- [ ] Increase the number of preset Agents
- [ ] Mixed retrieval, reordering and specified file retrieval
- [ ] 📚️Agent-driven Knowledge Base
