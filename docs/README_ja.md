# RAGENT
[中文文档](../docs/README_zh.md) | [English](../README.md) | **日本語**

おそらく最も軽量なローカル RAG + Agent アプリの一つで、複雑な設定なしに、ワンクリックで Agent の強化を受けた異なるモデルの能力の飛躍的な向上を体験できます。

> このプロジェクトが気に入ったら、ぜひ star を付けてください。それは私にとってとても重要です！

<img src="https://github.com/user-attachments/assets/bcc6395a-92ab-4ae6-8d36-a6831c240b16" alt="image" style="zoom: 50%;" />

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

#### レコードエクスポート

エクスポートフォーマット、テーマ選択、エクスポート範囲の制御をサポート：

<img src="https://github.com/user-attachments/assets/85756a3c-7ca2-4fcf-becc-682f22091c4e" alt="エクスポート設定及びプレビュー" style="zoom:40%;" />

現在サポートしているテーマ：

| クラシック | Glassmorphism |
| :----------: | :--------------: |
| <img src="https://github.com/user-attachments/assets/20a817f7-9fb9-4e7a-8840-f3072a39053a" alt="中文导出原皮" width="300" /> | <img src="https://github.com/user-attachments/assets/9fdc60ac-6eda-420c-ba7a-9e9bc97d8dcf" alt="中文导出透明皮" width="300" /> |

### RAG Chat

<img src="https://github.com/user-attachments/assets/bc574d1e-e614-4310-ad00-746c5646963a" alt="image" style="zoom:50%;" />
モデル設定（サイドバー） 及び 詳細引用の確認：

<img src="https://github.com/user-attachments/assets/a6ce3f0b-3c8f-4e3d-8d34-bceb834da81e" alt="image" style="zoom:50%;" />

RAG の設定：

<img src="https://github.com/user-attachments/assets/82480174-bac1-47d4-b5f4-9725774618f2" alt="image" style="zoom: 50%;" />

### Function Call（ツール呼び出し）

現在はChatでの関数呼び出しがサポートされており、将来的にはAgentChatでも関数呼び出しがサポートされる予定です。

ネイティブ呼び出しで、すべての OpenAI Compatible モデルに有効ですが、モデル自体が Function call をサポートする必要があります。

Function Call は LLM の能力を大幅に強化し、本来は不可能な作業（例：数学計算）を完了できるようにします。以下に例を示します：

<img src="https://github.com/user-attachments/assets/fba30f4a-dbfc-47d0-9f1c-4443171fa018" alt="Chat_page_tool_call zh_cn" style="zoom:50%;" />

またはウェブページの内容を要約する：

<img src="https://github.com/user-attachments/assets/7da5ae4d-40d5-49b4-9e76-6ce2a39ac6d1" alt="Chat_page_tool_call zh_cn" style="zoom:50%;" />

> あなた自身で呼び出したい関数をカスタマイズすることもできます。書き方は [toolkits.py](../tools/toolkits.py) の記述規則を参照してください。

## クイックスタート

### Git

0. `git clone https://github.com/Wannabeasmartguy/RAGENT.git`を使用してコードを取得します。
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

コマンドラインで：`streamlit run RAGENT.py` を実行すると起動し、起動完了後ブラウザで自動的にフロントエンドページが開きます。

コントリビュート
使用中に遭遇した問題や新しいアイデアについては、issue や PR の提出を歓迎します！