import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# --- パス設定 ---
# プロジェクトルートをsys.pathに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モック ---
# Streamlitやその他の重いライブラリをモック化し、インポートテストを軽量化
mock_streamlit = MagicMock()
mock_langchain = MagicMock()
mock_langgraph = MagicMock()
mock_networkx = MagicMock()

@pytest.fixture(scope="module", autouse=True)
def mock_heavy_libs():
    """テスト実行中に重いライブラリのインポートを無効化する"""
    with patch.dict('sys.modules', {
        'streamlit': mock_streamlit,
        'langchain': mock_langchain,
        'langchain.prompts': MagicMock(),
        'langchain.chains': MagicMock(),
        'langchain.text_splitter': MagicMock(),
        'langchain_community': mock_langchain,
        'langchain_community.document_loaders': MagicMock(),
        'langchain_community.llms': MagicMock(),
        'langchain_core': mock_langchain,
        'langchain_core.prompts': MagicMock(),
        'langchain_core.output_parsers': MagicMock(),
        'langchain_core.runnables': MagicMock(),
        'langchain_core.documents': MagicMock(),
        'langchain_huggingface': MagicMock(),
        'langchain_ollama': mock_langchain,
        'langchain_chroma': mock_langchain,
        'langgraph': mock_langgraph,
        'langgraph.graph': MagicMock(),
        'networkx': mock_networkx,
        'torch': MagicMock(),
        'chromadb': MagicMock(),
    }):
        yield

# --- テストケース ---

def test_modular_rag_imports():
    """
    ModularRAGがshared_componentsを正しく利用できるかテストする。
    （最初は失敗することを期待）
    """
    # このテストは、ModularRAG/app.pyがshared_componentsを
    # 正しくインポートできれば成功する。
    # app.py内で 'from components.pdf_processor import PDFProcessor' のような
    # 古いパスが使われているため、最初はImportErrorになるはず。
    # それを修正するのがゴール。
    try:
        from ModularRAG import app as modular_rag_app
        # PDFProcessorの初期化などをモックで乗り切る
        with patch('shared_components.pdf_processor.PDFProcessor'), \
             patch('shared_components.model_loader.load_llm.load_llm'):
            # モジュールレベルのコードが実行されることを確認
            assert modular_rag_app is not None
    except ImportError as e:
        pytest.fail(f"ModularRAG failed to import necessary components. Check paths. Error: {e}")

def test_rag_plus_imports():
    """
    RAG_plusがshared_componentsを正しく利用できるかテストする。
    """
    try:
        from RAG_plus import app as rag_plus_app
        with patch('shared_components.pdf_processor.PDFProcessor'), \
             patch('shared_components.model_loader.load_llm.load_llm'):
            assert rag_plus_app is not None
    except ImportError as e:
        pytest.fail(f"RAG_plus failed to import necessary components. Check paths. Error: {e}")

def test_rrarag_imports():
    """RRARagのインポートをテスト"""
    try:
        from RRARag import rrarag
        assert rrarag is not None
    except ImportError as e:
        pytest.fail(f"RRARag failed to import. Error: {e}")

def test_dynamicrag_imports():
    """DynamicRagのインポートをテスト"""
    try:
        from DynamicRag import DynaRag
        assert DynaRag is not None
    except ImportError as e:
        pytest.fail(f"DynamicRag failed to import. Error: {e}")

def test_mrdrag_imports():
    """MRD_Ragのインポートをテスト"""
    try:
        from MRD_Rag import mrdrag
        assert mrdrag is not None
    except ImportError as e:
        pytest.fail(f"MRD_Rag failed to import. Error: {e}")
