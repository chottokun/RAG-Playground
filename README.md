# RAGを実験的に実装

本リポジトリは、PDF文書を知識ベースとして活用し、検索拡張生成（RAG）の論文を参考に実装の実験を試験的に行っています。実装の学習目的であり、論文に記載の実装を忠実に再現したり、評価するものではありません。各論文に基づいた実装は論文や著者が直接公開しているソースコードを参照してください。

---

## リスト

- DeepRag: 段階的なクエリ分解・検索・生成を行うRAGデモ
- RRARag: Rational Retrieval Acts (RRA) に基づく再ランキングRAGデモ
- AgenticRag: LangGraphを活用したエージェント型RAGデモ
- DynamicRag: LLMのフィードバックによる動的な文書選択・再ランキングRAGデモ
- MADAM-RAG (MRD_Rag): 複数エージェントによるディベートと集約を行うRAGデモ

---

## アーキテクチャと開発ガイド

新しいRAG実装の追加方法、共通コンポーネントの使い方、テストの書き方など、開発に関する詳細は [Architecture.md](Architecture.md) を参照してください。

## ディレクトリ構成

```
AgenticRag/
  app.py              # AgenticRagデモ本体 (UI)
  graph_builder.py    # LangGraphのグラフ構築ロジック
  nodes/              # 各エージェントノードのコンポーネント
    retrieval.py
    ...
  config.ini          # AgenticRag設定ファイル
  README.md           # AgenticRagの詳細説明
DeepRag/
  app.py              # DeepRAGデモ本体 (UI)
  orchestrator.py     # DeepRAGのコアロジック
  components/         # DeepRAG固有のコンポーネント
    decomposition.py
    ...
  config.ini          # DeepRag設定ファイル
  README.md           # DeepRAGの詳細説明
... (他のRAG実装) ...
shared_components/    # 複数のRAG実装で共有される共通コンポーネント
  pdf_processor.py    # PDF処理・VectorDB管理
  model_loader/       # LLM・埋め込みモデルのロード
tests/                # テストコード
  test_deeprag.py
  test_agenticrag_nodes.py
  ...
development/          # 開発用ディレクトリ
  status.md
pdfs/                 # PDFファイル配置用ディレクトリ
.gitignore            # Git無視設定
guidelines.md         # RAGシステム共通仕様
Pipfile               # 依存パッケージ管理
README.md             # 本ファイル
```

---

## 各アプリの概要

### DeepRag
- PDFから知識ベースを構築し、質問をサブクエリに分解しながら段階的に検索・生成を行うRAGデモです。
- 詳細・使い方・フローは `DeepRag/README.md` を参照してください。

### RRARag
- Rational Retrieval Acts (RRA) アルゴリズムを用いて、ベクトル検索後の文書を再ランキングし、より高精度な検索拡張生成を実現する実装を簡略的に行いました。
- 詳細・使い方・フローは `RRARag/README.md` を参照してください。

### AgenticRag
- LangGraphの最新APIと共通モジュールを活用したエージェント型RAGシステムです。PDFから知識ベースを自動構築し、ユーザーの質問に対して段階的な検索・評価・クエリリファイン・最終回答生成を行います。
- 詳細・使い方・フローは `AgenticRag/README.md` を参照してください。

### DynamicRag
- LLMの出力をフィードバックとして活用し、ベクトル検索後の候補文書から本当に必要な文書のみを動的に選択・再ランキングするRAGデモです。
- 詳細・使い方・フローは `DynamicRag/README.md` を参照してください。

### MADAM-RAG (MRD_Rag)
- 複数エージェントが異なる役割・温度で文書から根拠を抽出し、アグリゲータが集約・説明する多段ディベート＆集約型RAGデモです。
- 詳細・使い方・フローは `MRD_Rag/README.md` を参照してください。

---

## 使い方

各RAGデモのセットアップ、設定ファイル (`config.ini`) の編集、実行方法、および詳細な使い方は、各デモディレクトリ内の `README.md` を参照してください。

一般的には以下の手順となります。

1. 各デモディレクトリ内の `config.ini` を編集し、使用するLLMやPDFファイル、ベクトルストアディレクトリなどを設定します。
2. 各デモの **`app.py`** ファイルをStreamlitで起動します。例: `streamlit run DeepRag/app.py`
3. アプリのUIからPDFのインデックス作成や質問入力を行います。

### テストの実行

本プロジェクトでは`pytest`によるテストが整備されています。

```bash
# すべてのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_deeprag.py
```

---

## 依存パッケージ
- streamlit
- langchain
- langgraph
- langchain-ollama
- langchain-chroma
- langchain-huggingface
- langchain_community
- chromadb
- sentence-transformers
- torch, configparser など
- **pytest** (開発用)

---

## 情報/Infomation
- 本リポジトリは実装技術勉強を個人で行っているものです。論文の検証や実装を追究するためではありません。
- This repository is used for personal study of implementation techniques. It is not intended for verifying or pursuing the implementation of research papers.

##
Thank you for conducting various research studies and publishing your papers.
