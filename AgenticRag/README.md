# Agentic RAG Demo

本アプリケーションは、LangGraphを用いて構築されたエージェント型RAG（Retrieval-Augmented Generation）のデモです。ユーザーの質問に対し、検索・評価・クエリ改善・再検索・回答生成という一連の思考プロセスをエージェント（ノード）の連携によって実行します。

本実装は、[AGENTIC RETRIEVAL-AUGMENTED GENERATION](https://arxiv.org/abs/2501.09136)などのサーベイ論文で議論されているエージェント型RAGのコンセプトを参考に、モジュール性とテスト容易性を重視して構築されています。

## 特徴
- **LangGraphベースのフロー制御**: 各処理ステップをノードとして定義し、`StateGraph`で柔軟な実行フローを構築しています。
- **モジュール化設計**: UI (`app.py`)、グラフ定義 (`graph_builder.py`)、ノード実装 (`nodes/`) が明確に分離されています。
- **共通コンポーネント利用**: PDF処理やモデルロードは、リポジトリ共通の`shared_components`を利用しています。
- **テストカバレッジ**: `tests/`ディレクトリに各ノードの単体テストが整備されており、各コンポーネントの動作が保証されています。

---

## ファイル構成
```
AgenticRag/
  ├── app.py              # UI (Streamlit)
  ├── graph_builder.py    # LangGraphのStateGraphを構築するロジック
  ├── nodes/              # グラフの各ノード（エージェント）の実装
  │   ├── __init__.py
  │   ├── retrieval.py
  │   ├── evaluation.py
  │   ├── refinement.py
  │   └── synthesis.py
  └── config.ini          # 設定ファイル
```

---

## 処理フロー

```mermaid
graph TD
    subgraph app.py (UI)
        A[質問入力] --> B{実行ボタン};
        B -- invoke --> C[graph.invoke(state)];
        C -- result --> D[結果表示];
    end

    subgraph graph_builder.py
        G["StateGraph定義"]
        G --> N1["retriever_node"];
        N1 --> N2["evaluator_node"];
        N2 --> N3["refiner_node"];
        N3 --> N4["refined_retriever_node"];
        N4 --> N5["synthesizer_node"];
    end

    subgraph nodes
        NODE1["retrieval.py"];
        NODE2["evaluation.py"];
        NODE3["refinement.py"];
        NODE4["synthesis.py"];
    end

    C --> G;
    N1 --> NODE1;
    N2 --> NODE2;
    N3 --> NODE3;
    N5 --> NODE4;
```
1.  **`app.py`**がユーザー入力を受け取り、初期状態(`AgenticState`)を作成して`graph.invoke`を呼び出します。
2.  **`graph_builder.py`**で定義されたグラフが実行を開始します。
3.  グラフは`retriever` -> `evaluator` -> `refiner` -> `refined_retriever` -> `synthesizer`の順で、**`nodes/`**ディレクトリに実装された各ノード関数を呼び出します。
4.  各ノードは状態を更新し、最終的に`synthesizer`が生成した回答が`app.py`に返却され、UIに表示されます。

---

## コード解説

### `app.py`
- Streamlitを使用してUIを構築します。
- `shared_components`と`graph_builder`をインポートします。
- `get_components`関数で、`PDFProcessor`と`LLM`のインスタンスをキャッシュ付きでロードします。
- `build_agentic_graph()`を呼び出して、実行可能なLangGraphのグラフを構築します。
- ユーザーからの質問をトリガーに、初期状態を作成し`graph.invoke()`を実行して、結果を整形して表示します。

### `graph_builder.py`
- `AgenticState`という`TypedDict`を定義し、グラフ内でり受け渡しされるデータの型を定めます。
- `nodes`ディレクトリから各ノード関数をインポートします。
- `StateGraph`をインスタンス化し、`.add_node()`で各関数をノードとして登録し、`.add_edge()`でそれらの実行順序を定義します。
- `.compile()`で実行可能なグラフオブジェクトを生成し、返す`build_agentic_graph`関数を公開します。

### `nodes/`
- **`retrieval.py`**: 初期検索と再検索を行うノード。
- **`evaluation.py`**: 検索結果の関連性を評価するノード。
- **`refinement.py`**: 評価結果に基づき、検索クエリを改善するノード。
- **`synthesis.py`**: 最終的な検索結果から回答を生成するノード。
- 各ファイルは、`state`を引数に取り、更新された`state`の一部を辞書として返す、という規約に従います。

---

## 使い方

1.  `AgenticRag/config.ini`を編集し、モデル名やベクトルストアのディレクトリを設定します。
2.  `pdfs/`ディレクトリに、知識ベースとしたいPDFファイルを配置します。
3.  以下のコマンドでアプリケーションを起動します。
    ```bash
    streamlit run AgenticRag/app.py
    ```
4.  ブラウザにUIが表示されたら、質問を入力し「実行」ボタンを押します。インデックスが存在しない場合は、初回実行時に自動で作成されます。

## 設定例（config.ini）
```ini
[LLM]
PROVIDER = ollama

[ollama]
BASE_URL = http://localhost:11434
MODEL = gemma3:4b-it-qat

[embedding]
MODEL = intfloat/multilingual-e5-small

[vectorstore]
DIRECTORY = ./vectorstore_agenticrag

[pdf]
# pdfs/ ディレクトリ全体を対象とするため、個別のPATH指定は不要
DIRECTORY = ./pdfs
```
