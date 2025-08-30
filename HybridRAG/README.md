# HybridRAG: Vector + Knowledge Graph RAG

このアプリケーションは、従来のベクトル検索（VectorRAG）とナレッジグラフ検索（GraphRAG）を組み合わせた**HybridRAG**のアプローチを実装したデモです。

金融文書のような専門的で構造的な情報を含むテキストから、より正確で文脈に即した情報を抽出することを目的とします。

## 特徴
- **ハイブリッド検索**:
    - **VectorRAG**: テキスト全体の意味的な類似性に基づき、広範な関連情報を取得します。
    - **GraphRAG**: テキストから抽出したエンティティと関係性に基づくナレッジグラフを探索し、構造化された情報を取得します。
- **コンテキストの結合**: VectorRAGとGraphRAGの両方から得られたコンテキストを結合し、LLMに渡すことで、よりリッチな情報に基づいた回答生成を可能にします。
- **モジュール化設計**: UI、Orchestrator、コンポーネントが分離されており、拡張やメンテナンスが容易です。

## ファイル構成
```
HybridRAG/
  ├── app.py              # UI (Streamlit)
  ├── orchestrator.py     # HybridRAGのコアロジック
  ├── graph_builder.py    # テキストからナレッジグラフを構築するロジック
  ├── components/
  │   ├── graph_retrieval.py # グラフ検索コンポーネント
  │   └── synthesis.py     # 回答生成コンポーネント
  └── config.ini          # 設定ファイル
```
