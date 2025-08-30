import pytest
from unittest.mock import MagicMock, patch

# パス解決
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- テスト対象 ---
# これでlangchainのインポートは不要になったはず
from DeepRag.components.decomposition import decomposition_component
from DeepRag.components.synthesis import synthesize_intermediate_answer, synthesize_final_answer

# --- テストケース ---

def test_decomposition_component():
    """decomposition_componentがプロンプトをフォーマットしてLLMを呼び出すかテスト"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "  Sub Query 1  "

    # プロンプトテンプレートのモック
    mock_template = MagicMock()
    mock_template.format.return_value = "Formatted Decomposition Prompt"

    question = "What is RAG?"
    history = "prev_q ; prev_a"

    result = decomposition_component(mock_llm, mock_template, question, history)

    # テンプレートが正しくフォーマットされたか
    mock_template.format.assert_called_once_with(question=question, history=history)
    # LLMがフォーマット後のプロンプトで呼び出されたか
    mock_llm.invoke.assert_called_once_with("Formatted Decomposition Prompt")
    # 結果がstripされているか
    assert result == "Sub Query 1"

def test_synthesize_intermediate_answer():
    """synthesize_intermediate_answerがプロンプトをフォーマットしてLLMを呼び出すかテスト"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = " Intermediate Answer "

    mock_template = MagicMock()
    mock_template.format.return_value = "Formatted Synthesis Prompt"

    query = "Sub Query 1"
    context = "Document content here."

    result = synthesize_intermediate_answer(mock_llm, mock_template, query, context)

    mock_template.format.assert_called_once_with(query=query, context=context)
    mock_llm.invoke.assert_called_once_with("Formatted Synthesis Prompt")
    assert result == "Intermediate Answer"

def test_synthesize_final_answer():
    """synthesize_final_answerが中間回答を結合してLLMを呼び出すかテスト"""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "  Final Answer  "

    question = "What is RAG?"
    intermediate_answers = ["Answer 1.", "Answer 2."]

    result = synthesize_final_answer(mock_llm, question, intermediate_answers)

    expected_prompt = f"Based on above, answer: {question}\nContext: Answer 1.\n\nAnswer 2."
    mock_llm.invoke.assert_called_once_with(expected_prompt)
    assert result == "Final Answer"
