# RAGenT

**中文文档** | [English](../README.md) | [日本語](README_ja.md)

可能是最轻量级的本地 RAG + Agent 应用之一，无需进行复杂配置，即可一键体验 Agent 加持下不同模型能力的强大提升。

> 如果你喜欢这个项目，请为它点一个 star ，这对我很重要！

<img src="https://github.com/user-attachments/assets/bcc6395a-92ab-4ae6-8d36-a6831c240b16" alt="image" style="zoom: 50%;" />

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
  - [x] 多主题选择（HTML）

知识库：
- [x] **原生实现**的检索增强生成（RAG），轻量而高效
- [x] 可供选择的嵌入模型（Hugging Face/OpenAI）
- [x] 易于使用的知识库管理
- [x] 多种检索方法：混合检索、重排序和指定文件检索

> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 更多细节

### 通用

#### 记录导出

支持导出格式、主题选择和导出范围控制：

<img src="https://github.com/user-attachments/assets/85756a3c-7ca2-4fcf-becc-682f22091c4e" alt="导出设置及预览" style="zoom:40%;" />

目前已支持主题：

|                             经典                             |                        Glassmorphism                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://github.com/user-attachments/assets/20a817f7-9fb9-4e7a-8840-f3072a39053a" alt="中文导出原皮" width="300" /> | <img src="https://github.com/user-attachments/assets/9fdc60ac-6eda-420c-ba7a-9e9bc97d8dcf" alt="中文导出透明皮" width="300" /> |

### RAG Chat

<img src="https://github.com/user-attachments/assets/bc574d1e-e614-4310-ad00-746c5646963a" alt="image" style="zoom:50%;" />

设置模型（边栏）及查看详细引用：

<img src="https://github.com/user-attachments/assets/a6ce3f0b-3c8f-4e3d-8d34-bceb834da81e" alt="image" style="zoom:50%;" />

配置 RAG ：

<img src="https://github.com/user-attachments/assets/82480174-bac1-47d4-b5f4-9725774618f2" alt="image" style="zoom: 50%;" />

### Function Call（工具调用）

> 现在支持在`Chat`上进行函数调用，未来也将支持在`AgentChat`上进行函数调用。

原生调用，对所有 OpenAI Compatible 模型均有效，但需要模型本身支持 Function call。

Function Call 可以显著增强 LLM 的能力，使其完成原本无法完成的工作（如数学计算），如下所示：

<img src="https://github.com/user-attachments/assets/fba30f4a-dbfc-47d0-9f1c-4443171fa018" alt="Chat_page_tool_call zh_cn" style="zoom:50%;" />

或是总结网页内容：

<img src="https://github.com/user-attachments/assets/7da5ae4d-40d5-49b4-9e76-6ce2a39ac6d1" alt="Chat_page_tool_call zh_cn" style="zoom:50%;" />

> 你也可以自定义想要调用函数，编写请参考[toolkits.py](tools/toolkits.py)的编写规则。

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

   如果你想要使用 AgentChat 页面，请使用 `python startup.py` 启动应用，而不是 `streamlit run RAGenT.py`。

## 贡献

对于使用中遇到的问题和产生的新想法，欢迎提交 issue 和 PR！
