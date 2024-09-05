# RAGenT

**中文文档** | [English](README.md)

可能是最轻量级的本地 RAG + Agent 应用之一，无需进行复杂配置，即可一键体验 Agent 加持下不同模型能力的强大提升。

> 如果你喜欢这个项目，请为它点一个 star ，这对我很重要！

![image](https://github.com/user-attachments/assets/89638402-ca9d-4dda-94b2-cb40035cf543)

## 特点

聊天及 Agent 交互：
- [x] 💭 简洁易用的聊天框界面
- [x] 🌏️ 语言可选（简体中文、English）
- [x] 🎤 语音输入（可选，本地 whisper 提供 TTS 服务）
- [x] 🔧 多种（本地）模型来源的推理支持（OpenAI, Azure OpenAI, Groq, ollama, llamafile/OpenAI Like）
- [x] 原生 Function Call （OpenAI, Azure OpenAI, OpenAI Like, Ollama）
  - [x] 模型本身需支持 Function Call
- [x] 🤖 预置多种 Agent 模式（基于 AutoGen ）
- [x] 🖥️ 对话数据的本地存储和管

知识库：
- [x] **原生实现**的检索增强生成（RAG），轻量而高效
- [x] 可供选择的嵌入模型（Hugging Face/OpenAI）
- [x] 易于使用的知识库管理
- [x] 混合检索、重排序和指定文件检索

> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 更多细节

### 通用

语音输入

![image](https://github.com/user-attachments/assets/37ea413d-5ef6-4783-a2da-ed6d1d010f58)

### RAG Chat

![image](https://github.com/user-attachments/assets/54c254b6-44d6-4e38-a4f8-eebf1d826f75)

设置模型（边栏）及查看详细引用：

![image](https://github.com/user-attachments/assets/f825f2a4-8e26-4478-99fc-b8d18d33a4a7)

配置 RAG ：

![image](https://github.com/user-attachments/assets/53344031-0920-43f7-be2e-717119e4c081)

### Function Call（工具调用）

在 `Chat` 和 `AgentChat` 页面均支持 Function call，但实现方式不同。

#### Chat 页面

原生调用，对所有 OpenAI Compatible 模型均有效，但需要模型本身支持 Function call。

![Chat_page_tool_call zh_cn](https://github.com/user-attachments/assets/2a334fd7-e4e1-456d-bad7-7b463e2911d3)

> 你也可以自定义想要调用函数，编写请参考[toolkits.py](tools/toolkits.py)的编写规则。

#### AgentChat 页面

借由 AutoGen 框架实现（测试），对模型的兼容性请参考 [AutoGen](https://github.com/microsoft/autogen) 的文档。

Function call 可以显著增强 LLM 的能力，目前支持 OpenAI, Azure OpenAI, Groq 以及本地模型（[通过 LiteLLM + Ollama](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama#using-litellmollama-with-autogen)）。

![openai function call](https://github.com/user-attachments/assets/4eabcedb-5717-46b1-b2f4-4324b5f1fb67)

> 你也可以自定义想要调用函数，请注意 AutoGen 的函数编写与原生调用的函数**编写规则不同**，具体请参考[官方文档](https://microsoft.github.io/autogen/docs/tutorial/tool-use/)以及本项目的[tools.py](llm/aoai/tools/tools.py)。

## 快速开始

### Git

0. 使用`git clone https://github.com/Wannabeasmartguy/RAGenT.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 配置模型依赖项：修改 `.env_sample` 文件为 `.env` 并填写以下内容：

   - `LANGUAGE`: 支持`English`和`简体中文`，默认为`English`。
   - `AZURE_OAI_KEY` : 如果你使用 Azure OpenAI 模型，请在此处填写 api key；
   - `AZURE_OAI_ENDPOINT` : 如果你使用 OpenAI 模型，请在此处填写 end_point；
   - `API_VERSION`: 如果你使用 Azure OpenAI 模型，请在此处填写 api version；
   - `API_TYPE`: 如果你使用 Azure OpenAI 模型，请在此处填写 api type；
   - `GROQ_API_KEY` : 如果你使用 Groq 作为模型来源，请在此处填写 api key；
   - `COZE_ACCESS_TOKEN`: 如果你需要使用创建的 Coze Bot ，请在此处填写 access token；

> 使用 Llamafile 及其他 OpenAI-Like 模型（支持 OpenAI 接口）或本地 OpenAI API 服务，请在应用内设置 api key 和 endpoint ，并自行启动 Llamafile 等本地服务。

2. 启动应用：

   命令行运行：`python startup.py` 即可启动，启动完成后会在浏览器自动打开前端页面。

> 你可以通过设置 `FRONT_PORT` 和`SERVER_PORT` 来修改前端和后端端口，它们的默认值分别为 5998 和 8000 。

## Todo

- [x] 聊天记录及配置本地持久化
  - [x] 聊天记录本地持久化
  - [x] 配置本地持久化
- [ ] 增加对更多模型来源的支持
- [ ] 增加更多可供调用的函数工具
- [ ] 增加对更多嵌入模型的支持
- [ ] 增加预置 Agent 的数量

## 贡献

对于使用中遇到的问题和产生的新想法，欢迎提交 issue 和 PR！
