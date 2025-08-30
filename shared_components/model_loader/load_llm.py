import os
from langchain_community.llms import AzureOpenAI, OpenAI
from langchain_ollama import OllamaLLM

def _get_param(param_name, env_names, kwargs, default=None):
    """
    引数、環境変数、デフォルト値の優先順位でパラメータを取得する。
    1. kwargs[param_name] があれば最優先
    2. 環境変数にあれば次に優先
    3. どちらもなければdefault値を返す
    """
    # 1. 引数(kwargs)をチェック
    if param_name in kwargs and kwargs[param_name] is not None:
        return kwargs[param_name]

    # 2. 環境変数をチェック
    if isinstance(env_names, str):
        env_names = [env_names]
    for env in env_names:
        val = os.environ.get(env)
        if val is not None:
            return val

    # 3. デフォルト値を返す
    return default

def load_llm(provider: str, **kwargs):
    """
    provider: 'azure', 'openai', 'ollama' のいずれか
    kwargs: 各プロバイダーに必要なパラメータ（例: model, deployment_name, base_url など）
    環境変数優先で取得
    """
    if provider == "azure":
        return AzureOpenAI(
            deployment_name=_get_param("deployment_name", ["AZURE_OPENAI_DEPLOYMENT_NAME"], kwargs, "gpt-4"),
            openai_api_version=_get_param("openai_api_version", ["AZURE_OPENAI_API_VERSION"], kwargs, "2024-05-01-preview"),
            openai_api_key=_get_param("openai_api_key", ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"], kwargs),
            azure_endpoint=_get_param("azure_endpoint", ["AZURE_OPENAI_ENDPOINT"], kwargs),
        )
    elif provider == "openai":
        return OpenAI(
            model=_get_param("model", ["OPENAI_MODEL"], kwargs, "gpt-4-turbo"),
            openai_api_key=_get_param("openai_api_key", ["OPENAI_API_KEY"], kwargs),
        )
    elif provider == "ollama":
        return OllamaLLM(
            model=_get_param("model", ["OLLAMA_MODEL"], kwargs, "llama3:70b"),
            base_url=_get_param("base_url", ["OLLAMA_BASE_URL"], kwargs, "http://localhost:11434"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

# 使用例
if __name__ == "__main__":
    providers = [
        ("azure", {"deployment_name": "gpt-4", "openai_api_version": "2024-05-01-preview"}),
        ("openai", {"model": "gpt-4-turbo"}),
        ("ollama", {"model": "llama3:70b", "base_url": "http://localhost:11434"}),
    ]
    for provider, params in providers:
        model = load_llm(provider, **params)
        response = model.generate(["AIの未来について100字で説明せよ"])
        print(f"{provider.upper()}の回答:")
        print(response.generations[0][0].text)
        print("---")