# RAGenT

 [English](README.md) | [中文文档](README_zh.md) | **日本語**

おそらく最も軽量なネイティブRAG + Agentアプリの1つであり、複雑な設定なしで、ワンクリックでAgentによるモデルと知識ベースの強力な機能を体験できます。

![image](https://github.com/user-attachments/assets/f50c9b86-55c8-4881-a7cf-b2ccf3b35ece)

## 特徴

チャットとエージェントのインタラクション:
- [x] 💭 シンプルで使いやすいチャットボックスインターフェース
- [x] 🌏️ 言語オプション（簡体字中国語、英語）
- [x] 🔧 複数の（ローカル）モデルソースの推論サポート（Azure OpenAI、Groq、ollama、llamafile）
- [x] ネイティブのFunction Call（OpenAI、Azure OpenAI、OpenAI Like、Ollama）
- [x] 🤖 複数のエージェントモードをオンプレミスで提供
- [x] 🖥️ ダイアログデータのローカルストレージと管理
  - [x] 複数のエクスポート形式（Markdown、HTML）
  - [x] 複数のテーマ（HTML）

知識ベース:
- [x] **ネイティブ実装**のリトリーバル強化生成（RAG）、軽量で効率的
- [x] オプションの埋め込みモデル（Hugging Face/OpenAI）
- [x] 使いやすい知識ベース管理
- [x] ハイブリッド検索、再ランキング、および指定ファイルのリトリーバル

> このプロジェクトが気に入ったら、スターを付けてください。それが私にとって最大の励みです！

## 詳細

### 一般

#### 音声入力:

<img src="https://github.com/user-attachments/assets/37ea413d-5ef6-4783-a2da-ed6d1d010f58" alt="image" style="zoom:50%;" />

#### エクスポート

エクスポート形式、テーマ選択、エクスポート範囲の制御をサポートします。

<img src="https://github.com/user-attachments/assets/f4e1461e-2334-4b45-b4d9-2ff0d79a0e63" alt="Export settings and preview" style="zoom:40%;" />

現在サポートされているテーマ:

| デフォルト | ガラス化 |
| :-----: | :-----------: |
| <img src="https://github.com/user-attachments/assets/6ac8132c-0821-4487-9a1a-a0297a35783a" alt="default theme" width="300" /> | <img src="https://github.com/user-attachments/assets/87b07e86-dd98-4e66-a850-17b776fbeb1c" alt="Glassmorphism theme" width="300" /> |



### RAGチャット

モデルの設定（サイドバー）と詳細な参照の表示:

<img src="https://github.com/user-attachments/assets/4fba2259-3362-42b2-a4d5-85e0658d7720" alt="image" style="zoom:50%;" />

RAGの設定：

<img src="https://github.com/user-attachments/assets/565d96dc-3f42-4f7d-a368-55af9f4a5d77" alt="image" style="zoom:50%;" />

### Function Call

Function Callは`Chat`と`AgentChat`の両方のページでサポートされていますが、実装方法が異なります。

#### チャットページ

このページのFunction Callはネイティブであり、すべてのOpenAI互換モデルで動作しますが、モデル自体がFunction Callをサポートしている必要があります。

<img src="https://github.com/user-attachments/assets/75163c4d-bcd2-4ef0-83d5-ab27c6527715" alt="image" style="zoom:50%;" />

> 呼び出したい関数をカスタマイズすることもできます。記述ルールについては[toolkits.py](tools/toolkits.py)を参照してください。

#### エージェントチャットページ

AutoGenフレームワークに依存して実装されています（テスト中）。モデルの互換性については[AutoGen](https://github.com/microsoft/autogen)のドキュメントを参照してください。

Function CallはLLMの能力を大幅に強化することができ、現在はOpenAI、Azure OpenAI、Groq、およびローカルモデルをサポートしています。（[LiteLLM + Ollama](https://microsoft.github.io/autogen/docs/topics/non-openai-models/local-litellm-ollama#using-litellmollama-with-autogen)による）

<img src="https://github.com/user-attachments/assets/4eabcedb-5717-46b1-b2f4-4324b5f1fb67" alt="openai function call" style="zoom:50%;" />

> 呼び出したい関数をカスタマイズすることもできます。AutoGenの関数記述はネイティブの関数呼び出し記述ルールとは**異なる**ことに注意してください。詳細については[公式ドキュメント](https://microsoft.github.io/autogen/docs/tutorial/tool-use/)およびこのプロジェクトの[tools.py](llm/aoai/tools/tools.py)を参照してください。

## クイックスタート

### Git

0. `git clone https://github.com/Wannabeasmartguy/RAGenT.git`を使用してコードを取得します。
次に、**コマンドプロンプト（CMD）**で実行環境を開き、`pip install -r requirements.txt`を使用して実行依存関係をインストールします。

1. モデル依存関係を設定します：`.env_sample`ファイルを`.env`に変更し、以下の内容を記入します：

    - `LANGUAGE`: `English`と`简体中文`をサポートし、デフォルトは`English`です。
    - `OPENAI_API_KEY` : OpenAIモデルを使用している場合、ここにAPIキーを記入します。
    - `AZURE_OAI_KEY` : Azure OpenAIモデルを使用している場合、ここにAPIキーを記入します。
    - `AZURE_OAI_ENDPOINT` : OpenAIモデルを使用している場合、ここにエンドポイントを記入します。
    - `API_VERSION`: Azure OpenAIモデルを使用している場合、ここにAPIバージョンを記入します。
    - `API_TYPE`: Azure OpenAIモデルを使用している場合、ここにAPIタイプを記入します。
    - `GROQ_API_KEY` : Groqをモデルソースとして使用している場合、ここにAPIキーを記入します。
    - `COZE_ACCESS_TOKEN`: 作成したCoze Botを使用する必要がある場合、ここにアクセストークンを記入します。

> Llamafileを使用している場合、Llamafileモデルを起動した後、アプリケーション内でエンドポイントを設定してください。

2. アプリケーションを起動します：

コマンドラインで`streamlit run RAGenT.py`を実行すると起動します。

## ルート

- [x] チャット履歴と設定のローカル永続化
    - [x] チャット履歴のローカル永続化
    - [x] 設定のローカル永続化
- [ ] プリセットエージェントの数を増やす
- [ ] 混合リトリーバル、再ランキング、および指定ファイルのリトリーバル
- [ ] 📚️エージェント駆動の知識ベース
