# from langchain.prompts import PromptTemplate # 削除

def decomposition_component(llm, prompt_template, question: str, history_str: str) -> str:
    """
    LLMを使って質問をサブクエリに分解するコンポーネント。

    Args:
        llm: 使用する言語モデルのインスタンス。
        prompt_template: 使用するプロンプトテンプレート。
        question: 分解対象の質問。
        history_str: これまでのやり取りの履歴。

    Returns:
        生成されたサブクエリ、または "TERMINATE"。
    """
    prompt = prompt_template.format(question=question, history=history_str)
    sub_query = llm.invoke(prompt).strip()
    return sub_query
