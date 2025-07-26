# RAGシステム開発ガイドライン

## 1. 概要

本ドキュメントは、本リポジトリにおけるRetrieval Augmented Generation (RAG) システムの標準アーキテクチャ、ディレクトリ構成、および開発プラクティスを定義します。このガイドラインに従うことで、様々なRAG手法の実装・比較・拡張を効率的かつ高品質に行うことを目指します。

## 2. 標準アーキテクチャ

各RAG実装は、以下のコンポーネントで構成されることを基本とします。

```mermaid
graph TD
    subgraph Shared Components
        SC1["shared_components/model_loader"]
        SC2["shared_components/pdf_processor"]
    end

    subgraph RAG Implementation (e.g., DeepRag)
        UI["app.py (Streamlit)"]
        CORE{"Orchestrator / Graph Builder"}
        COMP["Components / Nodes"]
    end

    subgraph Data
        DOC["Documents (PDFs)"]
        VS["Vectorstore"]
    end

    UI --> CORE;
    CORE --> COMP;
    CORE --> SC1;
    CORE --> VS;

    UI --> SC2;
    SC2 --> DOC;
    SC2 --> VS;
```

-   **UI (`app.py`)**: Streamlitによるユーザーインターフェース。ユーザーからの入力受付と結果表示を担当します。
-   **Orchestrator / Graph Builder (`orchestrator.py` or `graph_builder.py`)**: RAGのコアロジックを管理する中核。コンポーネントを呼び出し、全体の処理フローを制御します。
-   **Components / Nodes (`components/` or `nodes/`)**: RAGの各ステップを実装する独立したモジュール群（例：質問分解、検索、評価、回答生成など）。
-   **Shared Components (`shared_components/`)**: 全RAG実装で共通利用されるモジュール。
    -   `model_loader`: LLMや埋め込みモデルをロードします。
    -   `pdf_processor`: PDFの読み込み、チャンク化、Vectorstoreの作成・管理を行います。
-   **Documents / Vectorstore**: 入力ドキュメントと、それらをベクトル化したデータベース。

## 3. ディレクトリ構成

新しいRAG実装を追加する際は、以下のディレクトリ構成に従ってください。

```
(RAG名)/
  ├── app.py              # UI (Streamlit)
  ├── orchestrator.py     # または graph_builder.py などのコアロジック
  ├── components/         # または nodes/ などのコンポーネント群
  │   ├── __init__.py
  │   └── ...
  ├── config.ini          # 当該RAG実装固有の設定ファイル
  └── README.md           # 当該RAG実装の詳細な説明
```

## 4. 開発プロセスとテスト

品質を担保し、安全なリファクタリングを可能にするため、テスト駆動開発（TDD）を推奨します。

-   **テストディレクトリ**: すべてのテストコードは、リポジトリルートの`tests/`ディレクトリに配置します。
-   **テストフレームワーク**: `pytest`を使用します。
-   **テストの単位**:
    -   **コンポーネントテスト**: 各コンポーネント（`components/`や`nodes/`内のモジュール）が独立して正しく動作することを確認する単体テストを必ず作成してください。
    -   **統合テスト**: OrchestratorやGraph Builderが、各コンポーネントを正しく連携させて一連のフローを実行できるかを確認するテストも作成します。
-   **モッキング**: LLMのAPI呼び出しやファイルシステムへのアクセスなど、外部依存性を持つ部分は`unittest.mock`を用いて適切にモック化し、テストの安定性と速度を確保してください。

### テストの実行

```bash
# すべてのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_deeprag.py
```

## 5. 実行方法

各RAGアプリケーションは、それぞれの`app.py`を`streamlit`で実行します。

```bash
streamlit run DeepRag/app.py
```

## 6. 共通モジュールの利用

コードの重複を避け、保守性を高めるために、`shared_components`内のモジュールを積極的に利用してください。

-   **LLMのロード**: `shared_components.model_loader.load_llm`を使用します。
-   **PDF処理とVectorstore管理**: `shared_components.pdf_processor.PDFProcessor`を使用します。

これにより、前処理やモデルのインスタンス化に関するロジックを各実装で個別に書く必要がなくなります。
