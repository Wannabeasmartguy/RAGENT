# RAGENT

**English** | [中文文档](./docs/README_zh.md) | [日本語](./docs/README_ja.md)

Probably one of the lightest native RAG + Agent apps out there，experience the power of Agent-powered models and Agent-driven knowledge bases in one click, without complex configuration.

![image](https://github.com/user-attachments/assets/bcc6395a-92ab-4ae6-8d36-a6831c240b16)

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
- [x] Multiple search methods available: Hybrid search, reranking, and specified file retrieval

> If you like this project, please star it, it's the biggest encouragement for me!

## More details

### General

#### Export

Support export format, theme selection and export range control:

<img src="https://github.com/user-attachments/assets/85756a3c-7ca2-4fcf-becc-682f22091c4e" alt="Export settings and preview" style="zoom:40%;" />

Currently supported themes: 

| Default | Glassmorphism |
| :-----: | :-----------: |
| <img src="https://github.com/user-attachments/assets/6ac8132c-0821-4487-9a1a-a0297a35783a" alt="default theme" width="300" /> | <img src="https://github.com/user-attachments/assets/87b07e86-dd98-4e66-a850-17b776fbeb1c" alt="Glassmorphism theme" width="300" /> |

### RAG Chat

<img src="https://github.com/user-attachments/assets/bc574d1e-e614-4310-ad00-746c5646963a" alt="image" style="zoom:50%;" />

You can set up the model (sidebar) and view detailed references:

<img src="https://github.com/user-attachments/assets/a6ce3f0b-3c8f-4e3d-8d34-bceb834da81e" alt="image" style="zoom:50%;" />

Configure RAG：

<img src="https://github.com/user-attachments/assets/82480174-bac1-47d4-b5f4-9725774618f2" alt="image" style="zoom:50%;" />

### Function Call

> Function calls are supported on `Chat` now, and `AgentChat` will be supported in the future.

The Function Calls on this page are native and work for all OpenAI Compatible models, but require the model itself to support Function calls.

Function Call can significantly enhance the capabilities of LLM, allowing it to complete tasks that it was previously unable to complete (such as mathematical calculations), as shown below:

<img src="https://github.com/user-attachments/assets/fba30f4a-dbfc-47d0-9f1c-4443171fa018" alt="image" style="zoom:50%;" />

Or summarize the content of a webpage:

<img src="https://github.com/user-attachments/assets/7da5ae4d-40d5-49b4-9e76-6ce2a39ac6d1" alt="image" style="zoom:50%;" />

> You can also customize the function you want to call, please refer to [toolkits.py](tools/toolkits.py) for writing rules.

## Quick start

### Git

0. Use `git clone https://github.com/Wannabeasmartguy/RAGENT.git` to pull the code;
Then open your runtime environment in **command prompt (CMD)** and use `pip install -r requirements.txt` to install the runtime dependencies.

1. Configure the model dependencies: Modify the `.env_sample` file to `.env` and fill in the following:

    - `LANGUAGE`: Support `English` and `简体中文`, if not set, default is `English`;
    - `OPENAI_API_KEY` : If you are using an OpenAI model, fill in the api key here;
    - `AZURE_OAI_KEY` : If you are using an Azure OpenAI model, fill in the api key here;
    - `AZURE_OAI_ENDPOINT` : If you are using an OpenAI model, fill in the end_point here;
    - `API_VERSION`: If you are using an Azure OpenAI model, fill in the api version here;
    - `API_TYPE`: if you are using an Azure OpenAI model, fill in the api type here;
    - `GROQ_API_KEY` : if you are using Groq as the model source, fill in the api key here;
    - `COZE_ACCESS_TOKEN`: if you need to use the created Coze Bot, fill in the access token here;

> If you are using Llamafile, please set the endpoint within the application after starting the Llamafile model.

2. launch the application:

Run: Run `streamlit run RAGENT.py` on the command line can start it.

If you want to use the AgentChat page, please use `python startup.py` to start the application rather than `streamlit run RAGENT.py`.

## Contribution

For any issues encountered during use or new ideas, please submit issues and PRs!
