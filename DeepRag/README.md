# DeepRAG: Step-by-Step RAG Demo

このアプリケーションは、PDF文書を知識ベースとして活用し、段階的なクエリ分解と検索を行うRAG（Retrieval-Augmented Generation）システムのデモです。

本実装は、[DeepRAG: Thinking to Retrieval Step by Step](https://arxiv.org/abs/2502.01142)の論文のアイデアを参考に、モジュール化とテスト容易性を重視してリファクタリングされています。

## 特徴
- **モジュール化設計**: UI (`app.py`)、コアロジック (`orchestrator.py`)、個別機能 (`components/`) が分離されており、見通しが良く拡張しやすい構造です。
- **共通コンポーネント利用**: PDF処理やモデルロードは、リポジトリ共通の`shared_components`を利用しており、コードの重複を排除しています。
- **段階的クエリ分解（DeepRAG）**: 質問をサブクエリに分解し、各サブクエリごとに検索・回答を繰り返し、最終回答を生成します。
- **テストカバレッジ**: `tests/`ディレクトリに`pytest`で実行可能なテストコードが整備されており、品質が保証されています。

---

## ファイル構成
```
DeepRag/
  ├── app.py              # UI (Streamlit)
  ├── orchestrator.py     # DeepRAGのコアロジック（Orchestrator）
  ├── components/         # DeepRAG固有のコンポーネント
  │   ├── __init__.py
  │   ├── decomposition.py  # 質問分解コンポーネント
  │   └── synthesis.py      # 回答生成コンポーネント
  └── config.ini          # 設定ファイル
```

---

## 処理フロー

```mermaid
flowchart TD
    subgraph app.py (UI)
        A[質問入力] --> B{実行ボタン押下};
        B -- Yes --> C[Orchestrator実行];
        C --> D[結果表示];
    end

    subgraph orchestrator.py
        E[binary_tree_search 開始] --> F[decomposition_component呼び出し];
        F --> G[Vectorstoreで検索];
        G --> H[synthesis_component呼び出し];
        H --> I{TERMINATE?};
        I -- No --> F;
        I -- Yes --> J[最終回答生成];
    end

    subgraph components
        COMP1["decomposition.py"];
        COMP2["synthesis.py"];
    end

    C --> E;
    F --> COMP1;
    H --> COMP2;
    J --> C;
```

---

## コード解説

### `app.py`
- Streamlitを使用してUIを構築します。
- `shared_components`の`PDFProcessor`を呼び出してVectorstoreのインデックス作成・ロードを行います。
- `shared_components`の`model_loader`を呼び出してLLMをロードします。
- `orchestrator.py`の`DeepRAG`クラスをインスタンス化し、ユーザーの質問を渡して実行します。
- `orchestrator`から返された最終回答とトレース情報を画面に表示します。

### `orchestrator.py`
- `DeepRAG`クラスが定義されており、RAGの主要な処理フローを管理します。
- `__init__`でLLM、Vectorstore、および各処理で使用するプロンプトテンプレートを初期化します。
- `binary_tree_search`メソッド内で、再帰的に以下の処理を呼び出します。
    1.  `components.decomposition.decomposition_component`を呼び出し、質問をサブクエリに分解します。
    2.  得られたサブクエリでVectorstoreを検索します。
    3.  `components.synthesis.synthesize_intermediate_answer`を呼び出し、中間回答を生成します。
    4.  最終的に`synthesize_final_answer`で回答を統合します。

### `components/`
- **`decomposition.py`**: 質問を分解するためのプロンプトをフォーマットし、LLMを呼び出す責務を持ちます。
- **`synthesis.py`**: 検索結果から中間回答を生成、またはすべての中間回答から最終回答を生成するためのプロンプトをフォーマットし、LLMを呼び出す責務を持ちます。

---

## 使い方

1.  `DeepRag/config.ini`を編集し、モデル名やベクトルストアのディレクトリを設定します。
2.  `pdfs/`ディレクトリに、知識ベースとしたいPDFファイルを配置します。
3.  以下のコマンドでアプリケーションを起動します。
    ```bash
    streamlit run DeepRag/app.py
    ```
4.  ブラウザで表示されたUIのサイドバーにある「Index PDF」ボタンを押し、インデックスを作成します（初回のみ）。
5.  質問を入力し、「Run DeepRAG」ボタンを押して結果を確認します。

## 設定例（config.ini）
```ini
[llm]
PROVIDER = ollama
MODEL = gemma3:4b-it-qat

[ollama]
BASE_URL = http://localhost:11434

[embedding]
MODEL = intfloat/multilingual-e5-small

[vectorstore]
DIRECTORY = ./vectorstore_deeprag

[pdf]
# 注意: PATHではなくDIRECTORYを指定します
DIRECTORY = ./pdfs
```
