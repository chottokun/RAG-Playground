from langchain_huggingface import HuggingFaceEmbeddings

# シンプルな埋め込みモデルローダーのダミー実装
# 必要に応じて本物のモデルロード処理に置き換えてください

def load_embedding_model(model_name: str):
    """
    指定したモデル名で HuggingFaceEmbeddings を返す
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"trust_remote_code": True}
    )
