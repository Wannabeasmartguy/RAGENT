# RAGenT

**中文文档** | [English](README.md) | [日本語](README_ja.md)

可能是最轻量级的本地 RAG + Agent 应用之一，无需进行复杂配置，即可一键体验 Agent 加持下不同模型能力的强大提升。

> 如果你喜欢这个项目，请为它点一个 star ，这对我很重要！

<img src="https://github.com/user-attachments/assets/d87905e8-bbb8-4c36-baa6-2bcf95c882bb" alt="image" style="zoom: 50%;" />

## 特点

聊天及 Agent 交互：
- [x] 💭 简洁易用的聊天框界面
- [x] 🌏️ 语言可选（简体中文、English）
- [x] 🎤 语音输入（可选，本地 whisper 提供 TTS 服务）
- [x] 🔧 多种（本地）模型来源的推理支持（OpenAI, Azure OpenAI, Groq, ollama, llamafile/OpenAI Like）
- [x] 原生 Function Call （OpenAI, Azure OpenAI, OpenAI Like, Ollama）
  - [x] 模型本身需支持 Function Call
- [x] 🤖 预置多种 Agent 模式（基于 AutoGen ）
- [x] 🖥️ 对话数据的本地存储和管理
  - [x] 多格式导出支持（Markdown、HTML）
  - [x] 多主体选择（HTML）

知识库：
- [x] **原生实现**的检索增强生成（RAG），轻量而高效
- [x] 可供选择的嵌入模型（Hugging Face/OpenAI）
- [x] 易于使用的知识库管理
- [x] 混合检索、重排序和指定文件检索

> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 更多细节

### 通用

#### 语音输入

<img src="https://github.com/user-attachments/assets/37ea413d-5ef6-4783-a2da-ed6d1d010f58" alt="image" style="zoom:50%;" />

#### 记录导出

支持导出格式、主题选择和导出范围控制：

<img src="https://github.com/user-attachments/assets/744ddca3-5eef-4774-91e7-06dc46e89931" alt="导出设置及预览" style="zoom:40%;" />

目前已支持主题：

|                             经典                             |                        Glassmorphism                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/user-attachments/assets/20a817f7-9fb9-4e7a-8840-f3072a39053a" alt="中文导出原皮" width="300" /> | <img src="https://github.com/user-attachments/assets/9fdc60ac-6eda-420c-ba7a-9e9bc97d8dcf" alt="中文导出透明皮" width="300" /> |



### RAG Chat

设置模型（边栏）及查看详细引用：

<img src="https://github.com/user-attachments/assets/e9a4ffb3-72da-4dbf-b82a-4c47e81231f2" alt="image" style="zoom:50%;" />

配置 RAG ：

<img src="https://github.com/user-attachments/assets/ce26b34d-6620-4517-a008-02f35cde2588" alt="image" style="zoom: 50%;" />

### Function Call（工具调用）

在 `Chat` 和 `AgentChat` 页面均支持 Function call，但实现方式不同。

#### Chat 页面

原生调用，对所有 OpenAI Compatible 模型均有效，但需要模型本身支持 Function call。

<img src="https://github.com/user-attachments/assets/2a334fd7-e4e1-456d-bad7-7b463e2911d3" alt="Chat_page_tool_call zh_cn" style="zoom:50%;" />

> 你也可以自定义想要调用函数，编写请参考[toolkits.py](tools/toolkits.py)的编写规则。

#### AgentChat 页面

借由 AutoGen 框架实现（测试），对模型的兼容性请参考 [AutoGen](https://github.com/microsoft/autogen) 的文档。

Function call 可以显著增强 LLM 的能力，目前支持 OpenAI, Azure OpenAI, Groq 以及本地模型（[通过 LiteLLM + Ollama](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama#using-litellmollama-with-autogen)）。

<img src="https://github.com/user-attachments/assets/4eabcedb-5717-46b1-b2f4-4324b5f1fb67" alt="openai function call" style="zoom:50%;" />

> 你也可以自定义想要调用函数，请注意 AutoGen 的函数编写与原生调用的函数**编写规则不同**，具体请参考[官方文档](https://microsoft.github.io/autogen/docs/tutorial/tool-use/)以及本项目的[tools.py](llm/aoai/tools/tools.py)。

## 快速开始

### Git

0. 使用`git clone https://github.com/Wannabeasmartguy/RAGenT.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 配置模型依赖项：修改 `.env_sample` 文件为 `.env` 并填写以下内容：

   - `LANGUAGE`: 支持`English`和`简体中文`，默认为`English`。
   - `OPENAI_API_KEY` : 如果你使用 OpenAI 模型，请在此处填写 api key；
   - `AZURE_OAI_KEY` : 如果你使用 Azure OpenAI 模型，请在此处填写 api key；
   - `AZURE_OAI_ENDPOINT` : 如果你使用 OpenAI 模型，请在此处填写 end_point；
   - `API_VERSION`: 如果你使用 Azure OpenAI 模型，请在此处填写 api version；
   - `API_TYPE`: 如果你使用 Azure OpenAI 模型，请在此处填写 api type；
   - `GROQ_API_KEY` : 如果你使用 Groq 作为模型来源，请在此处填写 api key；
   - `COZE_ACCESS_TOKEN`: 如果你需要使用创建的 Coze Bot ，请在此处填写 access token；

> 使用 Llamafile 及其他 OpenAI-Like 模型（支持 OpenAI 接口）或本地 OpenAI API 服务，请在应用内设置 api key 和 endpoint ，并自行启动 Llamafile 等本地服务。

2. 启动应用：

   命令行运行：`streamlit run RAGenT.py` 即可启动，启动完成后会在浏览器自动打开前端页面。

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
