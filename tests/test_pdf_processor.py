import pytest
from unittest.mock import patch, MagicMock, mock_open

# パス解決のため、プロジェクトルートをsys.pathに追加
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モックの設定 ---
# 依存ライブラリをモック
mock_pypdf_loader = MagicMock()
mock_text_splitter = MagicMock()
mock_hf_embeddings = MagicMock()
mock_chroma = MagicMock()

# テスト用の設定ファイルの内容
mock_config_content = """
[pdf]
DIRECTORY = mock_pdfs/
[embedding]
MODEL = mock-embedding-model
[vectorstore]
DIRECTORY = mock_vectorstore/
"""

# --- Fixtureの定義 ---
@pytest.fixture
def mocked_pdf_processor(mock_langchain_dependencies):
    """
    依存ライブラリとファイルシステムをモックし、PDFProcessorクラスを返すFixture。
    """
    # configparserのopenをモックして、偽の設定ファイル内容を読み込ませる
    with patch("builtins.open", mock_open(read_data=mock_config_content)):
        # モックが有効なコンテキスト内でテスト対象をインポート
        from shared_components.pdf_processor import PDFProcessor

        # 各テスト実行前にモックをリセット
        # conftestでモックした実体をリセットする必要がある
        from unittest.mock import MagicMock
        # ここでリセットするのは難しいので、各テストケースでリセットするか、
        # conftestのfixture自体を関数スコープにするのが良い。
        # 今回はconftestのfixtureを関数スコープに変更するアプローチをとる。

        yield PDFProcessor(config_path='dummy/config.ini')

# --- テストケース ---

def test_init(mocked_pdf_processor):
    """PDFProcessorの初期化時に設定が正しく読み込まれるかテスト"""
    processor = mocked_pdf_processor
    assert processor.pdf_directory == "mock_pdfs/"
    assert processor.embedding_model_name == "mock-embedding-model"
    assert processor.persist_directory == "mock_vectorstore/"
    # HuggingFaceEmbeddingsが設定のモデル名で初期化されたか検証
    mock_hf_embeddings.assert_called_once_with(model_name="mock-embedding-model")

@patch("os.path.exists", return_value=True)
@patch("os.listdir", return_value=["doc1.pdf", "doc2.pdf", "other.txt"])
def test_index_pdfs_success(mock_listdir, mock_exists, mocked_pdf_processor):
    """PDFのインデックス作成が正常に行われるかテスト"""
    processor = mocked_pdf_processor

    # モックの設定
    # PyPDFLoader.load()が返すダミードキュメント
    mock_pypdf_loader.return_value.load.return_value = [MagicMock()]
    # RecursiveCharacterTextSplitter.split_documents()が返すダミーチャンク
    mock_splitter_instance = mock_text_splitter.return_value
    mock_splitter_instance.split_documents.return_value = [MagicMock(), MagicMock()] # 2 chunks per doc
    # Chroma.from_documents()が返すダミーストア
    mock_store = MagicMock()
    mock_chroma.from_documents.return_value = mock_store

    # 処理の実行
    result_store = processor.index_pdfs()

    # 検証
    assert mock_exists.called
    assert mock_listdir.called
    # PyPDFLoaderがPDFファイルごとに呼び出されたか
    assert mock_pypdf_loader.call_count == 2
    mock_pypdf_loader.assert_any_call(os.path.join("mock_pdfs/", "doc1.pdf"))
    mock_pypdf_loader.assert_any_call(os.path.join("mock_pdfs/", "doc2.pdf"))
    # TextSplitterが初期化されたか
    mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=100)
    # Chroma.from_documentsが合計4つのチャンクで呼び出されたか
    assert mock_chroma.from_documents.call_args[0][0] # 引数のリスト
    assert len(mock_chroma.from_documents.call_args[0][0]) == 4 # 2 docs * 2 chunks
    # 戻り値がChromaのインスタンスか
    assert result_store == mock_store

@patch("os.path.exists", return_value=False)
def test_index_pdfs_no_dir(mock_exists, mocked_pdf_processor):
    """PDFディレクトリが存在しない場合にNoneを返すかテスト"""
    processor = mocked_pdf_processor
    result = processor.index_pdfs()
    assert result is None
    # Chromaは呼び出されない
    mock_chroma.from_documents.assert_not_called()

@patch("os.path.exists", return_value=True)
def test_load_vectorstore_success(mock_exists, mocked_pdf_processor):
    """Vectorstoreのロードが正常に行われるかテスト"""
    processor = mocked_pdf_processor
    mock_store_instance = MagicMock()
    mock_chroma.return_value = mock_store_instance

    store = processor.load_vectorstore()

    # os.path.existsが永続化ディレクトリのパスで呼ばれたか
    mock_exists.assert_called_with("mock_vectorstore/")
    # Chromaが永続化ディレクトリとエンベディング関数を指定して初期化されたか
    mock_chroma.assert_called_once()
    assert mock_chroma.call_args[1]['persist_directory'] == "mock_vectorstore/"
    assert mock_chroma.call_args[1]['embedding_function'] is not None
    # 戻り値がChromaのインスタンスか
    assert store == mock_store_instance

@patch("os.path.exists", return_value=False)
def test_load_vectorstore_no_dir(mock_exists, mocked_pdf_processor):
    """Vectorstoreディレクトリが存在しない場合にNoneを返すかテスト"""
    processor = mocked_pdf_processor
    result = processor.load_vectorstore()
    assert result is None
    # Chromaは呼び出されない
    mock_chroma.assert_not_called()

def test_init_raises_error_for_nonexistent_config(mock_langchain_dependencies):
    """存在しないconfigパスを渡すとFileNotFoundErrorを送出するかのテスト"""
    from shared_components.pdf_processor import PDFProcessor

    # os.path.existsをモックして、ファイルが存在しないように見せかける
    with patch("os.path.exists", return_value=False), \
         patch("builtins.open", mock_open(read_data="")) as mock_file:
        # read_dataを空にすると、config.read()が0を返す
        mock_file.side_effect = FileNotFoundError # open自体がエラーを出すようにする
        with pytest.raises(FileNotFoundError):
            PDFProcessor(config_path='non/existent/path.ini')
