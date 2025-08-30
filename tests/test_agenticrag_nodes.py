import pytest
from unittest.mock import MagicMock, patch

# パス解決
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# テスト対象のノード関数
from AgenticRag.nodes.retrieval import retrieve_node, refined_retrieve_node
from AgenticRag.nodes.evaluation import evaluator_node
from AgenticRag.nodes.refinement import refiner_node
from AgenticRag.nodes.synthesis import synthesizer_node

# --- Fixtures ---

@pytest.fixture
def mock_state():
    """テスト用の基本的な状態（state）を返すFixture"""
    mock_doc = MagicMock()
    mock_doc.page_content = "Mocked document content."

    mock_vectordb = MagicMock()
    mock_vectordb.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "LLM Response" # llm.invoke("prompt") の呼び出しをモック

    mock_config = MagicMock()
    mock_config.getint.return_value = 10

    return {
        "question": "Initial Question",
        "vectordb": mock_vectordb,
        "llm": mock_llm,
        "config": mock_config,
        "retriever": ["Doc 1 content", "Doc 2 content"],
        "evaluator": ["Score: 5 - Very relevant", "Score: 1 - Not relevant"],
        "refiner": "Refined Question",
        "refined_retriever": ["Refined Doc 1 content"],
    }

# --- Node Tests ---

def test_retrieve_node(mock_state):
    result = retrieve_node(mock_state)
    mock_state["vectordb"].similarity_search.assert_called_once_with("Initial Question", k=10)
    assert "retriever" in result
    assert result["retriever"] == ["Mocked document content."]

def test_refined_retrieve_node(mock_state):
    result = refined_retrieve_node(mock_state)
    mock_state["vectordb"].similarity_search.assert_called_once_with("Refined Question", k=10)
    assert "refined_retriever" in result
    assert result["refined_retriever"] == ["Mocked document content."]

def test_evaluator_node(mock_state):
    result = evaluator_node(mock_state)
    # 2つのドキュメントに対してLLMが2回呼ばれる
    assert mock_state["llm"].invoke.call_count == 2
    assert "evaluator" in result
    assert len(result["evaluator"]) == 2
    assert result["evaluator"][0] == "LLM Response"

def test_refiner_node(mock_state):
    result = refiner_node(mock_state)
    mock_state["llm"].invoke.assert_called_once()
    # プロンプトに質問と評価結果が含まれているか（簡易チェック）
    prompt_arg = mock_state["llm"].invoke.call_args[0][0]
    assert "Initial Question" in prompt_arg
    assert "Score: 5" in prompt_arg
    assert "refiner" in result
    assert result["refiner"] == "LLM Response"

def test_synthesizer_node(mock_state):
    result = synthesizer_node(mock_state)
    mock_state["llm"].invoke.assert_called_once()
    prompt_arg = mock_state["llm"].invoke.call_args[0][0]
    assert "Refined Question" in prompt_arg
    assert "Refined Doc 1 content" in prompt_arg
    assert "final_answer" in result
    assert result["final_answer"] == "LLM Response"

import configparser

# --- Config Loading Test ---
def test_config_fallback(tmp_path):
    """設定ファイルでMODELが欠落している場合のフォールバックをテスト"""
    config_content = """
[LLM]
PROVIDER = ollama
[ollama]
BASE_URL = http://localhost:11434
[embedding]
MODEL = intfloat/multilingual-e5-small
[vectorstore]
DIRECTORY = ./vectorstore_agentucrag
[pdf]
PATH = 2502.01142v1.pdf
"""
    config_path = tmp_path / "config.ini"
    config_path.write_text(config_content)

    config = configparser.ConfigParser()
    config.read(config_path)

    # フォールバックが機能することをテスト
    model = config.get('ollama', 'MODEL', fallback='gemma3:4b-it-qat')
    assert model == 'gemma3:4b-it-qat'
