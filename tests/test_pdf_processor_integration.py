import pytest
import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configparser
from shared_components.pdf_processor import PDFProcessor
from langchain_core.documents import Document

@pytest.fixture
def temp_dir(tmp_path):
    # Create a temporary directory structure
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()

    # Create a dummy PDF file
    # (We can't create a real PDF, so we'll create a text file and mock the loader)
    dummy_pdf = pdf_dir / "dummy.pdf"
    dummy_pdf.write_text("This is a dummy PDF.")

    # Create a dummy config file
    config = configparser.ConfigParser()
    config['pdf'] = {'DIRECTORY': str(pdf_dir)}
    config['embedding'] = {'MODEL': 'intfloat/multilingual-e5-small'}
    config['vectorstore'] = {'DIRECTORY': str(vectorstore_dir)}
    config_path = tmp_path / "config.ini"
    with open(config_path, 'w') as f:
        config.write(f)

    return tmp_path

def test_pdf_processor_init(temp_dir):
    config_path = temp_dir / "config.ini"
    processor = PDFProcessor(config_path=str(config_path))
    assert processor.pdf_directory == str(temp_dir / "pdfs")
    assert processor.persist_directory == str(temp_dir / "vectorstore")

@patch("langchain_community.document_loaders.PyPDFLoader")
def test_get_chunks_from_pdfs(mock_pypdf_loader, temp_dir):
    mock_pypdf_loader.return_value.load.return_value = [Document(page_content="This is a dummy PDF.")]
    config_path = temp_dir / "config.ini"
    processor = PDFProcessor(config_path=str(config_path))
    chunks = processor.get_chunks_from_pdfs()
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)

@patch("langchain_community.document_loaders.PyPDFLoader")
def test_index_and_load_vectorstore(mock_pypdf_loader, temp_dir):
    mock_pypdf_loader.return_value.load.return_value = [Document(page_content="This is a dummy PDF.")]
    config_path = temp_dir / "config.ini"
    processor = PDFProcessor(config_path=str(config_path))

    # Index the PDFs
    vectorstore = processor.index_pdfs()
    assert vectorstore is not None

    # Load the vectorstore
    loaded_vectorstore = processor.load_vectorstore()
    assert loaded_vectorstore is not None
