# RAG+

## 概要

RAG+は、従来のRAG（Retrieval-Augmented Generation）が得意とする「知識の検索」に加え、実際の応用例(Application-Aware Reasoning)も同時に提示することで、知識の活用までを支援する新しい枠組みです。本プログラムでは、PDFから知識と応用例の2種類のコーパスを自動構築し、ユーザーの質問に対して根拠となる知識と、その知識がどのように使われるかの具体例を同時に提示します。これにより、単なる情報検索にとどまらず、実践的な活用までを見据えた応答が可能となります。

本システムは、StreamlitベースのUIで、PDFドキュメントから知識・応用例のデュアルコーパスを構築し、ユーザーの質問に対して根拠（知識）と応用例の両方を提示します。

## 参考文献と本プログラムの狙い
[RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning](https://arxiv.org/abs/2506.11555)  

Yu Wang, Shiwan Zhao, Ming Fan, Zhihu Wang, Yubo Zhang, Xicheng Zhang, Zhengfan Wang, Heyuan Huang, Ting Liu

本プログラムは上記論文のアイデア・アルゴリズムを参考に実装されています。しかしながら、理論の評価や利用を目的としたものではなく、実装練習として一部の機能を実装しています。

## 特徴

- PDFから知識コーパス・応用例コーパスを自動構築（デュアルコーパス）
- 質問に対し、知識とその応用例を同時に検索・提示
- Application-Aware Reasoningによる高精度な回答生成
- 柔軟なコンポーネント設計（クエリ分解、リランキング、評価、合成など）
- 設定ファイル（config.ini）によるカスタマイズ

## ディレクトリ構成

```
RAG_plus/
  app.py                # Streamlitアプリ本体
  config.ini            # 設定ファイル
  components/           # 各種RAG+コンポーネント
  orchestrator.py       # 全体オーケストレーション
  evaluator.py          # 評価用
  shared_types.py       # 型定義
  README.md             # 本ファイル
vectorstore_rag_plus_knowledge/   # 知識コーパスのベクトルストア
vectorstore_rag_plus_application/ # 応用例コーパスのベクトルストア
pdfs/                  # PDFファイル配置ディレクトリ
```

## セットアップ

1. 必要なPythonパッケージをインストール
   ```
   pip install -r requirements.txt
   ```
2. `config.ini` を編集し、各種パラメータを設定
3. `pdfs/` ディレクトリにPDFを配置
4. Streamlitアプリを起動
   ```
   streamlit run RAG_plus/app.py
   ```

## 使い方

1. サイドバーの「Index PDFs for RAG+」ボタンでPDFをインデックス化（デュアルコーパス構築）
2. 質問を入力し「Run RAG」ボタンで実行
3. 回答とともに、根拠（Knowledge）・応用例（Application）が表示されます

## 設定例（config.ini）

```ini
[LLM]
PROVIDER = ollama
MODEL = gemma3:4b-it-qat

[embedding]
MODEL = intfloat/multilingual-e5-small

[vectorstore]
KNOWLEDGE_DIRECTORY = ./vectorstore_rag_plus_knowledge
APPLICATION_DIRECTORY = ./vectorstore_rag_plus_application

[pdf]
PATH = pdfs/
```

## 参考

- 詳細な要件・設計思想は `docmuments/Application-Aware_Reasoning.md` を参照
- 各コンポーネントの詳細は `components/` 配下のファイルを参照
