# RAGenT

**中文文档** | [English](README.md)

无需进行复杂配置，即可一键体验 Agent 加持下模型能力的强大提升，并体验由 Agent 驱动的知识库。

## 特点

聊天及 Agent 交互：
- [x] 💭 简洁易用的聊天框界面
- [x] 🌏️ 语言可选（简体中文、English）
- [x] 🔧 多种（本地）模型来源的推理支持（Azure OpenAI, Groq, ollama, llamafile）
- [x] 🤖 预置多种 Agent 模式

知识库：
- [x] 可供选择的嵌入模型（Hugging Face/OpenAI）
- [x] 易于使用的知识库管理
- [ ] 混合检索、重排序和指定文件检索
- [ ] 📚️Agent 驱动的知识库

> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 更多细节

一般聊天页：

![common chat in RAGenT](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/11f61cad-81eb-48f3-8e03-bab5a4bc9470)

Agent 对话：

![agentchat-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e2cf62b3-447e-4b85-96dd-0bdd1e9e822f)

可以查看具体的思考内容：

![agentchat_expand_thought-en](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/ed33578e-e463-4eb6-996a-786c0d517eb3)

## 快速开始

### Git

0. 使用`git clone https://github.com/Wannabeasmartguy/RAGenT.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 配置模型依赖项：修改 `.env_sample` 文件为 `.env` 并填写以下内容：

`AZURE_OAI_KEY` : 如果你使用 Azure OpenAI 模型，请在此处填写 api key；
`AZURE_OAI_ENDPOINT` : 如果你使用 OpenAI 模型，请在此处填写 end_point；
`GROQ_API_KEY` : 如果你使用 Groq 作为模型来源，请在此处填写 api key；

> 使用 Llamafile 请在启动 Llamafile 模型后，在应用内设置 endpoint。

2. 启动应用：

命令行运行：`streamlit run RAGenT.py` 即可启动。

## 路线

- [ ] 增加预置 Agent 的数量
- [ ] 混合检索、重排序和指定文件检索
- [ ] 📚️Agent 驱动的知识库