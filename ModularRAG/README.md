# ModularRAG

## 概要

ModularRAGは、モジュール化されたRetrieval-Augmented Generation (RAG)システムです。このシステムは、Streamlitを使用して構築されており、PDFドキュメントからの情報検索と質問応答を可能にします。柔軟性と拡張性を重視し、様々なRAGコンポーネントを組み合わせることで、複雑な質問にも対応できます。

## 機能

*   PDFドキュメントのインデックス作成
*   質問応答
*   モジュール化されたRAGコンポーネント（クエリ分解、検索、評価、洗練、リフレクション、マルチエージェントディベート、ランキング、合成）
*   設定ファイルによる柔軟なカスタマイズ

## 設定

設定ファイル `config.ini` を編集して、システムの動作をカスタマイズできます。

```ini
[LLM]
PROVIDER = ollama
MODEL = gemma3:4b-it-qat

[ollama]
BASE_URL = http://localhost:11434

[embedding]
MODEL = intfloat/multilingual-e5-small

[vectorstore]
DIRECTORY = ./vectorstore_modular

[pdf]
PATH = pdfs/

# Optional component-specific configurations
[retrieval]
K = 5

[reranking]
METHOD = llm
TOP_K = 5 # Number of documents after reranking

[debate]
NUM_AGENTS = 4
MAX_ROUNDS = 2

[query_decomposition]
ENABLE_DYNAMIC_DECOMPOSITION = True

[synthesis]
max_tokens = 512
temperature = 0.7
```

*   `LLM`: 使用するLLMプロバイダーとモデルを設定します。
*   `ollama`: Ollamaプロバイダーの設定（ベースURL）。
*   `embedding`: 使用する埋め込みモデルを設定します。
*   `vectorstore`: ベクトルストアのディレクトリを設定します。
*   `pdf`: PDFファイルのパスを設定します。
*   その他のセクション: 各RAGコンポーネントのパラメータを設定します。

## 使用方法

1.  **PDFの準備**: `pdfs/` ディレクトリにPDFファイルを配置します。
2.  **アプリケーションの実行**: 以下のコマンドを実行してStreamlitアプリケーションを起動します。

    ```bash
    streamlit run app.py
    ```

3.  **PDFのインデックス作成**: アプリケーションのサイドバーから「Index PDF」ボタンをクリックして、PDFドキュメントのインデックスを作成します。
4.  **質問の入力**: テキストエリアに質問を入力し、「Run RAG」ボタンをクリックして質問に対する回答を取得します。

## ファイル構造

```
ModularRAG/
├── README.md          # このファイル
├── app.py             # Streamlitアプリケーション
├── components/        # RAGコンポーネント
│   ├── pdf_processor.py # PDF処理コンポーネント
│   └── ...
├── config.ini         # 設定ファイル
├── evaluator.py       # クエリ評価モジュール
├── orchestrator.py    # RAGオーケストレーター
├── shared_types.py    # 共有型定義
└── ...
```

## 依存関係

*   streamlit
*   configparser
*   torch
*   langchain
*   ... (他の依存関係もここに記載)

※ 依存関係は、Pipfileまたはrequirements.txtを参照して正確に記述してください。

