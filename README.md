# RAGを実験的に実装

本リポジトリは、PDF文書を知識ベースとして活用し、検索拡張生成（RAG）の論文を参考に実装を試験的に行っています。論文に記載の実装を忠実に再現したり、評価するものではありません。

---

## リスト

- DeepRag: 段階的なクエリ分解・検索・生成を行うRAGデモ
- RRARag: Rational Retrieval Acts (RRA) に基づく再ランキングRAGデモ

---

## ディレクトリ構成

```
DeepRag/
  deeprag.py      # DeepRAGデモ本体
  config.ini      # 共通設定ファイル
  2502.01142v1.pdf # サンプルPDF
  README.md       # DeepRAGの詳細説明

RRARag/
  rrarag.py       # RRA RAGデモ本体
  README.md       # RRARagの詳細説明

vectorstore/      # ベクトルストア（自動生成）
README.md         # 本ファイル
```

---

## 各アプリの概要

### DeepRag
- PDFから知識ベースを構築し、質問をサブクエリに分解しながら段階的に検索・生成を行うRAGデモです。
- 詳細・使い方・フローは `DeepRag/README.md` を参照してください。

### RRARag
- Rational Retrieval Acts (RRA) アルゴリズムを用いて、ベクトル検索後の文書を再ランキングし、より高精度な検索拡張生成を実現するデモです。
- 詳細・使い方・フローは `RRARag/README.md` を参照してください。

---

## 共通の使い方
1. `DeepRag/config.ini` でPDFパスやモデル名、ベクトルストアディレクトリなどを設定します。
2. 各ディレクトリでStreamlitアプリを起動します。
   ```bash
   cd DeepRag && streamlit run deeprag.py
   # または
   cd RRARag && streamlit run rrarag.py
   ```
3. 初回は「Index PDF」ボタンでインデックスを作成してください。
4. 質問を入力し、各種RAGの挙動を体験できます。

---

## 依存パッケージ
- streamlit
- langchain-ollama
- langchain-chroma
- langchain-huggingface
- langchain_community
- chromadb
- torch, configparser など

---

## 参考
- 本リポジトリは技術検証・デモ用途を想定しています。

