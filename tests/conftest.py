import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="function") # or just @pytest.fixture
def mock_langchain_dependencies():
    """
    langchainとその他の重いライブラリの依存関係をモックする、共有Fixture。
    """
    mock_streamlit = MagicMock()
    mock_langchain = MagicMock()
    mock_langgraph = MagicMock()
    mock_networkx = MagicMock()

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
        # pdf_processorのテストで使うモックもここに追加
        'langchain_community.document_loaders.PyPDFLoader': MagicMock(),
        'langchain.text_splitter.RecursiveCharacterTextSplitter': MagicMock(),
        'langchain_huggingface.HuggingFaceEmbeddings': MagicMock(),
        'langchain_chroma.Chroma': MagicMock(),
    }) as patched_modules:
        yield patched_modules
