import pytest
from unittest.mock import MagicMock, patch
import networkx as nx
from langchain_core.documents import Document

# パス解決
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# テスト対象
from HybridRAG.graph_builder import KnowledgeGraphBuilder
from HybridRAG.components.graph_retrieval import graph_retrieval_component, _find_entities_in_query

# --- Fixtures ---

@pytest.fixture
def mock_llm():
    """知識抽出用のLLMモック"""
    llm = MagicMock()
    # 呼び出し順序に応じて異なるトリプレットを返すように設定
    llm.invoke.side_effect = [
        '[("Apple", "is a", "company"), ("Tim Cook", "is the CEO of", "Apple")]',
        '[("Google", "is known for", "Search Engine"), ("Sundar Pichai", "is CEO of", "Google")]',
        '[]' # 3回目以降の呼び出しは空
    ]
    return llm

@pytest.fixture
def mock_config():
    """設定のモック"""
    config = MagicMock()
    config.get.return_value = 'mock_graph.gml'
    return config

# --- KnowledgeGraphBuilder Tests ---

def test_extract_triplets(mock_llm, mock_config):
    builder = KnowledgeGraphBuilder(mock_llm, mock_config)
    text = "Some text about Apple."
    triplets = builder._extract_triplets(text)
    mock_llm.invoke.assert_called_once()
    assert triplets == [("Apple", "is a", "company"), ("Tim Cook", "is the CEO of", "Apple")]

def test_extract_triplets_security():
    """evalの脆弱性を確認するテスト"""
    malicious_string = "os.system('echo vulnerable')"
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = f"__import__('os').system('echo vulnerable')"
    mock_config = MagicMock()
    builder = KnowledgeGraphBuilder(mock_llm, mock_config)

    # ast.literal_evalは安全なため、コマンドは実行されずに例外が発生し、
    # _extract_tripletsは空のリストを返すはず
    result = builder._extract_triplets("some text")
    assert result == []

def test_build_from_documents(mock_llm, mock_config):
    builder = KnowledgeGraphBuilder(mock_llm, mock_config)
    docs = [
        Document(page_content="Text about Apple."),
        Document(page_content="Text about Google.")
    ]
    graph = builder.build_from_documents(docs)

    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 4
    assert "Apple" in graph.nodes
    assert "Google" in graph.nodes
    assert graph.has_edge("Tim Cook", "Apple")
    assert graph.edges[("Tim Cook", "Apple")]['label'] == "is the CEO of"

# --- Graph Retrieval Component Tests ---

@pytest.fixture
def sample_graph():
    """テスト用のサンプルグラフ"""
    G = nx.Graph()
    G.add_edge("Apple", "Tim Cook", label="has CEO")
    G.add_edge("Apple", "iPhone", label="produces")
    G.add_edge("Microsoft", "Satya Nadella", label="has CEO")
    return G

def test_find_entities_in_query(sample_graph):
    query = "Tell me about Apple and its CEO"
    entities = _find_entities_in_query(query, sample_graph)
    # "apple"と"ceo"はノード名にないが、"Apple"はマッチする
    assert len(entities) == 1
    assert "Apple" in entities

def test_find_entities_case_insensitivity(sample_graph):
    query = "tell me about apple"
    entities = _find_entities_in_query(query, sample_graph)
    assert len(entities) == 1
    assert "Apple" in entities # 元のケースが返されることを確認

def test_graph_retrieval_component(sample_graph):
    query = "What is Apple?"
    context = graph_retrieval_component(query, sample_graph)

    # 期待されるコンテキストが両方含まれているか
    # 順序は不定なので、両方の可能性をチェック
    expected1 = "Apple has CEO Tim Cook. Apple produces iPhone."
    expected2 = "Apple produces iPhone. Apple has CEO Tim Cook."

    assert context == expected1 or context == expected2

def test_graph_retrieval_no_entities(sample_graph):
    query = "Tell me about Google."
    context = graph_retrieval_component(query, sample_graph)
    assert "No relevant entities" in context
