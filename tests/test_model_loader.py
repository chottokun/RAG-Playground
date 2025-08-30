import pytest
from unittest.mock import patch, MagicMock

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# --- パス設定 ---
# プロジェクトルートをsys.pathに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- モックの準備 ---
# これらは、実際のlangchainライブラリがインポートされる前に設定する必要がある
mock_azure = MagicMock()
mock_openai = MagicMock()
mock_ollama = MagicMock()

# --- Fixtureの定義 ---
@pytest.fixture
def mocked_load_llm():
    """
    langchainライブラリをモックし、load_llm関数をインポートして返すFixture。
    このFixtureをテストケースで使うことで、モックが適用された状態でテストを実行できる。
    """
    with patch.dict('sys.modules', {
        'langchain_community.llms.azure': MagicMock(AzureOpenAI=mock_azure),
        'langchain_community.llms.openai': MagicMock(OpenAI=mock_openai),
        'langchain_ollama': MagicMock(OllamaLLM=mock_ollama), # langchain_ollama全体をモック
        # モジュール自体もモックする
        'langchain_community.llms': MagicMock(AzureOpenAI=mock_azure, OpenAI=mock_openai),
    }):
        # モックが有効なコンテキスト内でテスト対象をインポート
        from shared_components.model_loader.load_llm import load_llm

        # 各テスト実行前にモックをリセット
        mock_azure.reset_mock()
        mock_openai.reset_mock()
        mock_ollama.reset_mock()

        yield load_llm

# --- テストケース ---

def test_load_llm_ollama_with_args(mocked_load_llm):
    """引数でOllama LLMを正常にロードできるかテスト"""
    model_name = "test_llama"
    base_url = "http://test-host:1234"

    llm = mocked_load_llm(provider="ollama", model=model_name, base_url=base_url)

    mock_ollama.assert_called_once_with(model=model_name, base_url=base_url)
    assert llm == mock_ollama.return_value

@patch.dict(os.environ, {"OLLAMA_MODEL": "env_llama", "OLLAMA_BASE_URL": "http://env-host:5678"})
def test_load_llm_ollama_with_env_vars(mocked_load_llm):
    """環境変数でOllama LLMを正常にロードできるかテスト"""
    llm = mocked_load_llm(provider="ollama")

    mock_ollama.assert_called_once_with(model="env_llama", base_url="http://env-host:5678")
    assert llm == mock_ollama.return_value

def test_load_llm_openai_with_args(mocked_load_llm):
    """引数でOpenAI LLMを正常にロードできるかテスト"""
    api_key = "test_api_key"
    model_name = "test_gpt"

    llm = mocked_load_llm(provider="openai", model=model_name, openai_api_key=api_key)

    mock_openai.assert_called_once_with(model=model_name, openai_api_key=api_key)
    assert llm == mock_openai.return_value

@patch.dict(os.environ, {"OPENAI_API_KEY": "env_api_key", "OPENAI_MODEL": "env_gpt"})
def test_load_llm_openai_with_env_vars(mocked_load_llm):
    """環境変数でOpenAI LLMを正常にロードできるかテスト"""
    llm = mocked_load_llm(provider="openai")

    mock_openai.assert_called_once_with(model="env_gpt", openai_api_key="env_api_key")
    assert llm == mock_openai.return_value

def test_load_llm_azure_with_args(mocked_load_llm):
    """引数でAzure OpenAI LLMを正常にロードできるかテスト"""
    params = {
        "deployment_name": "test_deployment",
        "openai_api_version": "2025-01-01",
        "openai_api_key": "azure_key",
        "azure_endpoint": "https://test.azure.com"
    }
    llm = mocked_load_llm(provider="azure", **params)

    mock_azure.assert_called_once_with(**params)
    assert llm == mock_azure.return_value

@patch.dict(os.environ, {
    "AZURE_OPENAI_DEPLOYMENT_NAME": "env_deployment",
    "AZURE_OPENAI_API_VERSION": "2025-01-02",
    "AZURE_OPENAI_API_KEY": "env_azure_key",
    "AZURE_OPENAI_ENDPOINT": "https://env.azure.com"
})
def test_load_llm_azure_with_env_vars(mocked_load_llm):
    """環境変数でAzure OpenAI LLMを正常にロードできるかテスト"""
    llm = mocked_load_llm(provider="azure")

    expected_params = {
        "deployment_name": "env_deployment",
        "openai_api_version": "2025-01-02",
        "openai_api_key": "env_azure_key",
        "azure_endpoint": "https://env.azure.com"
    }
    mock_azure.assert_called_once_with(**expected_params)
    assert llm == mock_azure.return_value

def test_load_llm_unknown_provider(mocked_load_llm):
    """未知のプロバイダー名が与えられた場合にValueErrorを送出するかテスト"""
    with pytest.raises(ValueError, match="Unknown provider: no_such_provider"):
        mocked_load_llm(provider="no_such_provider")

@patch.dict(os.environ, {"OLLAMA_MODEL": "env_llama"})
def test_arg_precedence_over_env(mocked_load_llm):
    """引数が環境変数より優先されるかテスト"""
    llm = mocked_load_llm(provider="ollama", model="arg_llama", base_url="http://arg-host:1234")

    mock_ollama.assert_called_once_with(model="arg_llama", base_url="http://arg-host:1234")
    assert llm == mock_ollama.return_value
