# from langchain.prompts import PromptTemplate # 削除
from typing import List

def synthesize_intermediate_answer(llm, prompt_template, query: str, context: str) -> str:
    """
    検索コンテキストを使って中間回答を生成するコンポーネント。
    """
    prompt = prompt_template.format(query=query, context=context)
    answer = llm.invoke(prompt).strip()
    return answer

def synthesize_final_answer(llm, question: str, intermediate_answers: List[str]) -> str:
    """
    すべての中間回答を統合して最終的な回答を生成するコンポーネント。
    """
    final_context = "\n\n".join(intermediate_answers)
    # このプロンプトは単純なf-stringなので、テンプレートオブジェクトは不要
    prompt = f"Based on above, answer: {question}\nContext: {final_context}"
    final_answer = llm.invoke(prompt).strip()
    return final_answer
