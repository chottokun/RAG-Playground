## RAG+実装のための要件定義とアーキテクチャ設計

ご要望の通り、論文「RAG+: Enhancing Retrieval-Augmented Generation with Application-Aware Reasoning」に基づき、`langchain`と`chromadb`を利用してRAG+を実装するための要件、アーキテクチャ、および実装に必要な情報を詳細に解説します。

### 1\. RAG+のコンセプト（論文に基づく要約）

[cite\_start]RAG+は、従来のRetrieval-Augmented Generation（RAG）が「知識を検索すること」に重点を置くのに対し、「検索した知識をどのように適用・応用するか」という認知的なステップを明示的に組み込むことで、より高度な推論能力を実現するフレームワークです [cite: 2, 3]。

**中核的なアイデア:**

  * [cite\_start]**デュアルコーパス (Dual Corpus):** 事実情報を含む「知識コーパス」と、その知識がどのように利用されるかを示す「応用例コーパス」の2つを構築します [cite: 4, 26]。
  * [cite\_start]**応用を意識した推論 (Application-Aware Reasoning):** 推論時に、ユーザーの質問に関連する「知識」を検索するだけでなく、その知識に紐づけられた「応用例」も同時に取得します [cite: 5]。
  * [cite\_start]**プロンプトの強化:** 取得した「知識」と「応用例」の両方をプロンプトに含めることで、LLMが単に情報を参照するだけでなく、構造化された推論プロセスや問題解決のパターンを学び、より精度の高い回答を生成できるよう促します [cite: 25, 89]。

[cite\_start]このアプローチにより、特に数学、法律、医療といった専門的なドメインにおいて、標準的なRAGを大幅に上回る性能向上が報告されています [cite: 6]。

-----

### 2\. 実装に向けたシステム要件

RAG+を`langchain`と`chromadb`で実装するために、以下の要件を定義します。

#### **データ要件**

1.  **知識コーパス (Knowledge Corpus):**

      * システムの知識ベースとなるテキストデータの集合。
      * 各知識データには、一意のID（例: `knowledge_001`）が付与されている必要があります。
      * [cite\_start]論文では、定義、定理、事実情報などが含まれます [cite: 73, 348]。

2.  **応用例コーパス (Application Corpus):**

      * 知識コーパスの各項目が、実際にどのように使われるかを示す具体例の集合。
      * [cite\_start]例として、計算問題の解法プロセス、複数選択問題、判例などが挙げられます [cite: 76, 59, 95]。
      * 各応用例は、どの知識項目に対応するかが明確にマッピングされている必要があります。

3.  **知識と応用例のマッピング (Knowledge-Application Mapping):**

      * 「どの知識IDがどの応用例IDに対応するか」を定義するマッピング情報が必須です。
      * [cite\_start]このマッピングは、構築段階（オフライン）で事前に作成されます [cite: 66, 88][cite\_start]。1対多（1つの知識に複数の応用例）の関係も許容されます [cite: 84]。

#### **機能要件**

1.  **コーパス構築機能 (Offline Stage):**

      * 知識コーパスと応用例コーパスを`chromadb`に格納する機能。
      * 格納時、各ドキュメントに**メタデータ**として「ドキュメントタイプ（'knowledge' or 'application'）」と「知識ID」を付与する必要があります。これがアーキテクチャの鍵となります。
      * [cite\_start]論文によれば、応用例は既存のデータからマッチングさせるか、LLMを用いて自動生成します [cite: 69]。

2.  **推論パイプライン機能 (Online Stage):**

      * [cite\_start]**Step 1: 知識検索 (Knowledge Retrieval):** ユーザーからのクエリに基づき、`chromadb`の**知識コーパス**から関連性の高い知識ドキュメントを検索します [cite: 87]。
      * [cite\_start]**Step 2: 応用例検索 (Application Retrieval):** Step 1で得られた各知識ドキュメントのメタデータ（知識ID）を使い、`chromadb`の**応用例コーパス**から対応する応用例ドキュメントを**フィルタリング検索**で取得します [cite: 88]。
      * [cite\_start]**Step 3: プロンプト構築:** クエリ、検索された知識、検索された応用例を組み合わせ、LLMへの入力プロンプトを動的に生成します [cite: 89]。
      * [cite\_start]**Step 4: 回答生成:** 構築したプロンプトをLLMに渡し、最終的な回答を生成させます [cite: 90]。

-----

### 3\. アーキテクチャ設計

`langchain`と`chromadb`を用いたRAG+のアーキテクチャを以下に示します。

```
<< RAG+ Architecture Overview >>

+-----------------------------------------------------------------+
| OFFLINE: Construction Stage                                     |
+-----------------------------------------------------------------+
|                                                                 |
|  [Knowledge Data] ----> (Embed & Store) ----+                   |
|      - id: "k_001"                          |                   |
|      - text: "Bayes' Theorem is..."         |                   |
|                                             |                   |
|  [Application Data] --> (Embed & Store) --+ |                   |
|      - maps_to: "k_001"                   | |                   |
|      - text: "Question: A disease affects..| |                   |
|               Answer: We apply Bayes'..."  v v                   |
|                                    +--------------------------+ |
|                                    | ChromaDB Vector Store    | |
|                                    | - Doc 1 (metadata):      | |
|                                    |   { type: 'knowledge',   | |
|                                    |     id: 'k_001' }        | |
|                                    | - Doc 2 (metadata):      | |
|                                    |   { type: 'application', | |
|                                    |     knowledge_id: 'k_001'} |
|                                    +--------------------------+ |
+-----------------------------------------------------------------+

+-----------------------------------------------------------------+
| ONLINE: Inference Stage                                         |
+-----------------------------------------------------------------+
|                                                                 |
| [User Query] -> 1. Knowledge Retriever (LangChain)              |
|                   |                                             |
|                   | (Searches ChromaDB where type='knowledge')  |
|                   |                                             |
|                   +-> [Retrieved Knowledge Docs (e.g., k_001)]  |
|                                   |                             |
|         +-------------------------+                             |
|         |                                                       |
|         v 2. Application Retriever (Custom Logic)               |
|                   |                                             |
|                   | (For each knowledge doc, get 'id',          |
|                   |  search ChromaDB where type='application'   |
|                   |  and knowledge_id=id)                       |
|                   |                                             |
|                   +-> [Retrieved Application Docs]              |
|                                   |                             |
|         +-------------------------+                             |
|         |                         |                             |
|         v                         v                             |
|   3. Prompt Formatter -------> [Combined Prompt]                |
|                                   |                             |
|                                   v                             |
|                             4. LLM (LangChain)                  |
|                                   |                             |
|                                   v                             |
|                               [Final Answer]                    |
|                                                                 |
+-----------------------------------------------------------------+

```

#### **コンポーネント詳細**

  * **ChromaDB Vector Store:**

      * 知識ドキュメントと応用例ドキュメントの両方を格納します。
      * **メタデータ活用が核心です。** 各ドキュメントに `{ "type": "knowledge", "id": "k_001" }` や `{ "type": "application", "knowledge_id": "k_001" }` のようなメタデータを付与して格納します。これにより、知識の検索と、それに紐づく応用例の検索を効率的に行えます。

  * **Knowledge Retriever (知識検索器):**

      * `langchain.vectorstores.Chroma` と `langchain.retrievers` を使用して実装します。
      * 検索対象を知識コーパスに限定するため、`chromadb`のフィルタリング機能（`where={"type": "knowledge"}`）を利用してリトリーバーを初期化します。

  * **Application Retriever (応用例検索器):**

      * これは`langchain`の標準リトリーバーとは少し異なるカスタムロジックになります。
      * 知識検索器が返した知識ドキュメントのリストをループ処理します。
      * 各知識ドキュメントのメタデータから`id`を抽出し、その`id`を使って`chromadb`を再度フィルタリング検索（`where={"type": "application", "knowledge_id": "retrieved_id"}`）し、対応する応用例を取得します。

  * **Prompt Formatter (プロンプト整形器):**

      * `langchain.prompts.PromptTemplate` を使用します。
      * [cite\_start]論文のFigure 10, 13, 16で示されているように、知識と応用例を明確に区別してLLMに提示するテンプレートを設計します [cite: 468, 478, 485]。

    **プロンプトテンプレートの例:**

    ```python
    template = """
    Based on the following knowledge and its application examples, answer the user's question.

    ## Reference Knowledge:
    {knowledge}

    ## Application Example of the Knowledge:
    {application}

    ## User Question:
    {question}

    Answer:
    """
    ```

  * **LLM:**

      * `langchain`がサポートする任意のLLM（`langchain_openai`, `langchain_anthropic`など）を利用できます。

-----

### 4\. 実装ステップの詳細

#### **Step 1: データ準備とコーパス構築 (Offline)**

1.  **データ構造の定義:**
    知識と応用例のデータを準備します。各知識項目には一意なIDを振ります。

    ```python
    knowledge_data = [
        {"id": "k_math_01", "text": "Euler's Theorem for Homogeneous Functions: If P is a homogeneous polynomial of degree d, then..."},
        {"id": "k_law_01", "text": "Article 234: Whoever intentionally harms the body of another person shall be sentenced to..."}
    ]
    application_data = [
        {"knowledge_id": "k_math_01", "text": "Question: Let P(x, y) = x^3 + ... Use Euler's theorem... Correct Answer: ..."},
        {"knowledge_id": "k_law_01", "text": "Question: What criteria should be used...? Answer: C) Based on specific circumstances..."}
    ]
    ```

2.  **`chromadb`への格納:**
    `langchain`の`Chroma.from_texts`や`Chroma.add_texts`を使い、メタデータを付与してドキュメントを格納します。

    ```python
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embedding_function = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

    # 知識を格納
    vectorstore.add_texts(
        texts=[item["text"] for item in knowledge_data],
        metadatas=[{"type": "knowledge", "id": item["id"]} for item in knowledge_data],
        ids=[f"k_{i}" for i in range(len(knowledge_data))] # ChromaDB自体のID
    )

    # 応用例を格納
    vectorstore.add_texts(
        texts=[item["text"] for item in application_data],
        metadatas=[{"type": "application", "knowledge_id": item["knowledge_id"]} for item in application_data],
        ids=[f"a_{i}" for i in range(len(application_data))] # ChromaDB自体のID
    )
    ```

#### **Step 2: 推論パイプラインの実装 (Online)**

1.  **知識リトリーバーの作成:**
    `as_retriever`メソッドの`search_kwargs`でフィルタリングを指定します。

    ```python
    knowledge_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2, "filter": {"type": "knowledge"}}
    )
    ```

2.  **RAG+チェーンの構築:**
    `langchain`のRunnableインタフェース（LCEL）を使って、カスタムロジックを組み込みます。

    ```python
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate

    # 応用例を取得するカスタム関数
    def retrieve_applications(retrieved_knowledge_docs):
        applications = []
        for doc in retrieved_knowledge_docs:
            knowledge_id = doc.metadata.get("id")
            if knowledge_id:
                # knowledge_idを使って応用例をフィルタリング検索
                app_docs = vectorstore.similarity_search(
                    query=" ", # クエリはダミーで良い。フィルタリングが目的
                    k=1,
                    filter={"type": "application", "knowledge_id": knowledge_id}
                )
                if app_docs:
                    applications.append(app_docs[0].page_content)
        return "\n---\n".join(applications)

    # プロンプトテンプレートの定義
    prompt_template = PromptTemplate.from_template(
        """
        Answer the question based on the following context.

        [Knowledge]
        {knowledge}

        [Application Examples]
        {application}

        [Question]
        {question}
        """
    )

    # LLMの定義
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # LCELでチェーンを組み立て
    rag_plus_chain = (
        {
            "knowledge": (lambda x: x["question"]) | knowledge_retriever,
            "question": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            application=lambda x: retrieve_applications(x["knowledge"])
        )
        | {
            "question": lambda x: x["question"]["question"],
            "knowledge": lambda x: "\n---\n".join([doc.page_content for doc in x["knowledge"]]),
            "application": lambda x: x["application"]
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 実行
    question = "How many ways to choose 3 students from 6 boys and 4 girls, with at least one girl?"
    response = rag_plus_chain.invoke({"question": question})
    print(response)
    ```

-----

### 5\. 論文から得られるその他の重要な情報

  * [cite\_start]**応用例の生成方法:** 応用例コーパスがない場合、論文ではLLMを使って自動生成する手法が紹介されています [cite: 71][cite\_start]。知識を「概念的知識」と「手続き的知識」に分類し、それぞれに適したプロンプトで応用例（例: 多肢選択問題、演習問題）を生成します [cite: 73, 75, 79]。これは`langchain`のFew-shotプロンプトなどを活用して実装可能です。
  * [cite\_start]**モジュール性:** RAG+はモジュール性が高く、既存のRAGパイプラインに「プラグアンドプレイ」で機能強化として追加できると述べられています [cite: 35, 93]。今回のアーキテクチャも、既存のRAGに`Application Retriever`のロジックを追加する形で実現できます。
  * **リランキング:** より高度な実装として、Rerank RAG+があります 。一度検索した知識・応用例のペアを、再度LLMに評価させて最適なものを選択するアプローチです。
  * [cite\_start]**限界点:** 高品質な応用例コーパスの構築にはコストがかかること、知識と応用例のミスマッチが誤った推論を導く可能性があることなどが限界として挙げられています [cite: 210, 212]。実装時には、コーパスの品質管理が重要になります。

以上の要件定義、アーキテクチャ設計、および実装ステップに基づき、`langchain`と`chromadb`を用いてRAG+を構築することが可能です。成功の鍵は、**知識と応用例をメタデータで明確に紐づけて管理する**点にあります。

-----

### PDFからデュアルコーパスを構築する課題と戦略

既存のPDF（例：教科書、技術マニュアル、法律文書、学術論文）をRAG+のデータソースとして活用する上での最大の課題は、一つの文書内に混在する\*\*「知識（Knowledge）」**と**「応用例（Application）」\*\*をいかにして自動的に分離し、関連付けるかです。

**戦略の核心：**

1.  **構造的・意味的分割：** PDFを単なるテキストの羅列ではなく、意味のある塊（チャンク）に分割します。
2.  **分類とラベリング：** 各チャンクが「知識」か「応用例」かを分類します。
3.  **文脈に基づくマッピング：** 文書内での出現順序や近接性に基づいて、「知識」と「応用例」を紐付けます。

-----

### 具体的なステップ

このプロセスは、以下の4つの主要なステップで構成されます。

#### Step 1: PDFからのテキスト・構造抽出

まず、PDFからテキスト情報を抽出します。単純なテキスト抽出だけでなく、見出しや箇条書き、表などの構造情報を可能な限り維持することが後のステップで重要になります。

  * **推奨ツール：**
      * `PyPDF2`：基本的なテキスト抽出に使えますが、レイアウト情報は失われがちです。
      * `pdfplumber`：より高機能で、テキストの位置情報や表の構造を保ったまま抽出できるため、こちらを推奨します。
      * `pymupdf`：高速で高機能なライブラリです。

#### Step 2: テキストのチャンク化

抽出したテキストを、意味のある単位に分割（チャンク化）します。

  * **アイデア1：構造ベースのチャンク化（推奨）**

      * PDFの章（Chapter）、節（Section）、項（Item）などの見出しを区切りとしてチャンクを作成します。例えば、「1.1 Bayes' Theorem」や「Example 1.2」といった見出しで分割することで、文脈が保たれやすくなります。
      * `pdfplumber`などで抽出した見出しのフォントサイズやスタイルを手がかりに、分割ルールを実装できます。

  * **アイデア2：再帰的な文字ベースのチャンク化**

      * `langchain.text_splitter.RecursiveCharacterTextSplitter` を利用する方法。まずは段落（`\n\n`）、次に文（`.`）といったように、大きな区切りから順に分割を試みます。構造が明確でないPDFに有効です。

#### Step 3: チャンクの分類（知識 vs. 応用例）

このステップがRAG+構築の鍵となります。各チャンクに「knowledge」または「application」のラベルを付与します。

  * **アイデア1：ルールベース（キーワードマッチング）**

      * チャンクの先頭や内容に特定のキーワードが含まれているかで分類する、シンプルで高速な方法です。
      * **応用例キーワード：** `Example`, `Exercise`, `Problem`, `Case Study`, `Q&A`, `実践`, `演習問題`, `例題`, `ケーススタディ`, `問`
      * **知識キーワード：** `Definition`, `Theorem`, `Concept`, `Principle`, `Summary`, `概要`, `定義`, `定理`, `法則`

  * **アイデア2：LLMベース（ゼロショット分類）**

      * より柔軟で高精度な方法として、LLMに分類を任せます。`langchain`を使って簡単に実装できます。
      * **プロンプト例：**
        ```prompt
        You are an expert document analyst. Please classify the following text chunk into one of two categories: 'knowledge' or 'application'.

        - 'knowledge' refers to definitions, theorems, principles, factual descriptions, or explanations of concepts.
        - 'application' refers to worked examples, practical problems, case studies, or questions and answers that demonstrate how knowledge is used.

        Respond with only the category name ('knowledge' or 'application').

        Text Chunk:
        """
        {chunk_text}
        """

        Category:
        ```

#### Step 4: 知識と応用例のマッピング

分類されたチャンクを関連付け、最終的なデータセットを作成します。PDFのような線形的なドキュメントでは、\*\*「応用例は、その直前に現れた知識と関連している可能性が高い」\*\*というヒューリスティック（経験則）が非常に有効です。

  * **実装ロジック：**
    1.  チャンクを文書の出現順に処理します。
    2.  最後に現れた「knowledge」チャンクのIDを一時的に保存しておきます（`last_knowledge_id`）。
    3.  チャンクが「application」と分類された場合、そのチャンクのメタデータに `knowledge_id: last_knowledge_id` を追加してマッピングします。
    4.  チャンクが「knowledge」と分類された場合、`last_knowledge_id` をそのチャンクの新しいIDで更新します。
    5.  全てのチャンクに一意のIDを割り振ります（例：`uuid.uuid4()`を使用）。

-----

### PDFの種類に応じたアプローチ例

  * **教科書・技術書：**

      * 構造が明確なため、「構造ベースのチャンク化」と「ルールベースの分類」の組み合わせが効果的です。「例1.1」「問1.2」といったパターンを見つけやすいです。

  * **法律文書・判例集：**

      * [cite\_start]**知識：** 法律の条文（例：「刑法第234条」[cite: 354, 543]）。条文番号でチャンク化できます。
      * **応用例：** 上記条文が適用された具体的な判例。
      * この場合も構造が比較的明確なため、ルールベースのアプローチが有効です。

  * **学術論文：**

      * **知識：** 「Abstract」「Introduction」「Methods」セクションが該当します。
      * **応用例：** 「Results」「Discussion」「Case Study」セクションが該当します。
      * 構造は明確ですが、内容が複雑なため「LLMベースの分類」がより適している可能性があります。

### 実装コードのスケッチ

```python
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Step 1 & 2: PDFからチャンクを抽出 ---
loader = PyPDFLoader("your_document.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# --- Step 3: LLMベースの分類器を定義 ---
llm = ChatOpenAI(model="gpt-4o")
classification_prompt = PromptTemplate.from_template(
    """Classify the following text chunk as 'knowledge' or 'application'.
    'knowledge': Definitions, theorems, principles.
    'application': Worked examples, problems, case studies.
    Respond with only the category name.
    Text: "{chunk_text}"
    Category:"""
)
classifier_chain = classification_prompt | llm | StrOutputParser()

# --- Step 4: 分類とマッピング ---
knowledge_corpus = []
application_corpus = []
last_knowledge_id = None

for chunk in chunks:
    chunk_text = chunk.page_content
    category = classifier_chain.invoke({"chunk_text": chunk_text}).strip().lower()

    if category == "knowledge":
        # 新しい知識IDを生成
        knowledge_id = f"k_{uuid.uuid4()}"
        last_knowledge_id = knowledge_id # last_knowledge_idを更新
        
        knowledge_corpus.append({
            "doc_id": knowledge_id,
            "text": chunk_text,
            "metadata": {"type": "knowledge", "id": knowledge_id}
        })

    elif category == "application" and last_knowledge_id:
        # 応用例として、直前の知識IDと紐付ける
        application_id = f"a_{uuid.uuid4()}"
        
        application_corpus.append({
            "doc_id": application_id,
            "text": chunk_text,
            "metadata": {"type": "application", "knowledge_id": last_knowledge_id}
        })

# これで knowledge_corpus と application_corpus が完成
# あとはこれらをChromaDBに格納する
print(f"Found {len(knowledge_corpus)} knowledge chunks and {len(application_corpus)} application chunks.")
```

このプロセスを経ることで、非構造化されたPDFからRAG+の強力なデュアルコーパスを効率的に構築し、システムの推論能力を最大限に引き出す準備が整います。