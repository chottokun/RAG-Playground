# RAGを実験的に実装

本リポジトリは、PDF文書を知識ベースとして活用し、検索拡張生成（RAG）の論文を参考に実装の実験を試験的に行っています。実装の学習目的であり、論文に記載の実装を忠実に再現したり、評価するものではありません。各論文に基づいた実装は論文や著者が直接公開しているソースコードを参照してください。

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
- Rational Retrieval Acts (RRA) アルゴリズムを用いて、ベクトル検索後の文書を再ランキングし、より高精度な検索拡張生成を実現する実装を簡略的に行いました。
- 詳細・使い方・フローは `RRARag/README.md` を参照してください。

---

## 共通の使い方
1. `DeepRag/config.ini` でPDFパスやモデル名、ベクトルストアディレクトリなどを設定します。
2. Streamlitアプリを起動します。
3. 初回は「Index PDF」ボタンでインデックスを作成してください。
4. 質問を入力し、各種RAGの挙動を体験できます。

詳細は `DeepRag/README.md` や `RRARag/README.md` を参照してください。

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

## 情報/Infomation
- 本リポジトリは実装技術勉強を個人で行っているものです。論文の検証や実装を追究するためではありません。
- This repository is used for personal study of implementation techniques. It is not intended for verifying or pursuing the implementation of research papers.

##
Thank you for conducting various research studies and publishing your papers.