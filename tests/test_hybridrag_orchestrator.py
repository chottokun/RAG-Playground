import pytest
from unittest.mock import MagicMock, patch
import configparser

# パス解決
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HybridRAG.orchestrator import HybridRAGOrchestrator

@pytest.fixture
def mock_llm():
    """LLMのモック"""
    return MagicMock()

@pytest.fixture
def mock_vectorstore():
    """Vectorstoreのモック"""
    vectorstore = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Mocked document content."
    vectorstore.similarity_search.return_value = [mock_doc]
    return vectorstore

@pytest.fixture
def mock_knowledge_graph():
    """KnowledgeGraphのモック"""
    return MagicMock()

@pytest.fixture
def mock_config():
    """ConfigParserのモック"""
    config = MagicMock()
    config.getint.return_value = 10
    return config

def test_hybridrag_orchestrator_run(mock_llm, mock_vectorstore, mock_knowledge_graph, mock_config):
    """HybridRAGOrchestrator.runのテスト"""
    orchestrator = HybridRAGOrchestrator(
        llm=mock_llm,
        vectorstore=mock_vectorstore,
        knowledge_graph=mock_knowledge_graph,
        config=mock_config
    )

    query = "What is Apple?"
    orchestrator.run(query)

    mock_vectorstore.similarity_search.assert_called_once_with(query, k=10)
