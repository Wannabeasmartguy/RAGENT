# RAGenT

**English** | [中文文档](./docs/README_zh.md) | [日本語](./docs/README_ja.md)

Probably one of the lightest native RAG + Agent apps out there，experience the power of Agent-powered models and Agent-driven knowledge bases in one click, without complex configuration.

![image](https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMKZ5Bj6HCDNEk_vLk6cx5_GlFtd-wAArjAMRsUEYFU1f6bxbi4L58BAAMCAAN3AAM2BA.png)

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

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMRZ5CP8tOC8mZNa1Gfmn0ZrIi02rYAAj_BMRsUEYFUKE-qz7XnQGYBAAMCAAN3AAM2BA.png" alt="Export settings and preview" style="zoom:40%;" />

Currently supported themes: 

| Default | Glassmorphism |
| :-----: | :-----------: |
| <img src="https://github.com/user-attachments/assets/6ac8132c-0821-4487-9a1a-a0297a35783a" alt="default theme" width="300" /> | <img src="https://github.com/user-attachments/assets/87b07e86-dd98-4e66-a850-17b776fbeb1c" alt="Glassmorphism theme" width="300" /> |



### RAG Chat

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMLZ5Bl9d6RMZnIt6k1yonaBS9wXbcAArzAMRsUEYFUrbgUsfJJiMYBAAMCAAN3AAM2BA.png" alt="image" style="zoom:50%;" />

You can set up the model (sidebar) and view detailed references:

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMMZ5BmXto1f3yvbBPW4AsgwsRgg7UAAsPAMRsUEYFUY1YxLW9SCQgBAAMCAAN3AAM2BA.png" alt="image" style="zoom:50%;" />

Configure RAG ：

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMNZ5Bmr4RpbW6hLlk2vgABmgABTLpHzwACxMAxGxQRgVR4uuGHCePimwEAAwIAA3cAAzYE.png" alt="image" style="zoom:50%;" />

### Function Call

> Function calls are supported on `Chat` now, and `AgentChat` will be supported in the future.

The Function Calls on this page are native and work for all OpenAI Compatible models, but require the model itself to support Function calls.

Function Call can significantly enhance the capabilities of LLM, allowing it to complete tasks that it was previously unable to complete (such as mathematical calculations), as shown below:

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMOZ5CISmmhCaolgsanImFf0nEEL9QAA8ExGxQRgVTsFfNLpLODKQEAAwIAA3cAAzYE.png" alt="image" style="zoom:50%;" />

Or summarize the content of a webpage:

<img src="https://telegraph-image-4v7.pages.dev/file/AgACAgUAAyEGAASL2gZEAAMPZ5CMA6bEaIijxonoYAABA5HeGVJ1AAIXwTEbFBGBVFDdSgXTyOJ4AQADAgADdwADNgQ.png" alt="image" style="zoom:50%;" />

> You can also customize the function you want to call, please refer to [toolkits.py](tools/toolkits.py) for writing rules.

## Quick start

### Git

0. Use `git clone https://github.com/Wannabeasmartguy/RAGenT.git` to pull the code;
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

Run: Run `streamlit run RAGenT.py` on the command line can start it.

If you want to use the AgentChat page, please use `python startup.py` to start the application rather than `streamlit run RAGenT.py`.

## Contribution

For any issues encountered during use or new ideas, please submit issues and PRs!
