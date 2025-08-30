import pytest
from unittest.mock import MagicMock, patch

# パス解決
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モックの準備 ---
mock_prompt_template_class = MagicMock()
mock_llm_chain_class = MagicMock()

# --- Fixtureの定義 ---

@pytest.fixture
def mock_llm():
    """LLMのモックを作成するFixture"""
    llm = MagicMock()
    # LLMの呼び出しシーケンスを定義
    llm.invoke.side_effect = [
        "Subquery 1",
        "Intermediate Answer 1",
        "Subquery 2",
        "Intermediate Answer 2",
        "TERMINATE",
        "Final Answer"
    ]
    return llm

@pytest.fixture
def mock_vectorstore():
    """Vectorstoreのモックを作成するFixture"""
    vectorstore = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Mocked document content."
    vectorstore.similarity_search.return_value = [mock_doc]
    return vectorstore

@pytest.fixture
def deep_rag_instance(mock_langchain_dependencies, mock_llm, mock_vectorstore):
    """モックされた依存関係を使ってDeepRAGのインスタンスを生成するFixture"""
    # モックが有効な状態でインポート
    from DeepRag.orchestrator import DeepRAG

    # インスタンスを生成して返す
    return DeepRAG(llm=mock_llm, vectorstore=mock_vectorstore)


# --- テストケース ---

def test_binary_tree_search(deep_rag_instance, mock_llm, mock_vectorstore):
    """
    DeepRAG.binary_tree_searchの主要なロジックをテストする。
    """
    # Fixtureからインスタンスを取得
    deep_rag = deep_rag_instance

    # 実行
    question = "Initial Question"
    final_answer, trace = deep_rag.binary_tree_search(question, max_depth=5)

    # --- 検証 ---

    # 1. LLMの呼び出し回数
    # decomp (sub1, sub2, terminate) + answer (2回) + final_answer (1回)
    # invokeが6回呼ばれるはず
    assert mock_llm.invoke.call_count == 6

    # 2. Vectorstoreの呼び出し回数
    assert mock_vectorstore.similarity_search.call_count == 2
    mock_vectorstore.similarity_search.assert_any_call("Subquery 1", k=3)
    mock_vectorstore.similarity_search.assert_any_call("Subquery 2", k=3)

    # 3. トレースの内容
    assert len(trace) == 2
    assert trace[0]["subquery"] == "Subquery 1"
    assert trace[0]["answer"] == "Intermediate Answer 1"
    assert trace[1]["subquery"] == "Subquery 2"
    assert trace[1]["answer"] == "Intermediate Answer 2"

    # 4. 最終回答
    assert final_answer == "Final Answer"

    # 5. 最終回答生成時のLLM呼び出し
    final_context = "\n\n".join(["Intermediate Answer 1", "Intermediate Answer 2"])
    mock_llm.invoke.assert_called_with(f"Based on above, answer: {question}\nContext: {final_context}")
