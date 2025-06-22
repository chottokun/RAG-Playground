import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from model_loader and shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from model_loader.load_llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import shared types
from RAG_plus.shared_types import RAGState

# Prompt template for query refinement
REFINEMENT_PROMPT = PromptTemplate(
    template="""Given the original user query, the conversation history, and the current answer, refine the original query to improve the next retrieval attempt. Focus on clarifying ambiguities, adding necessary context, or narrowing down the scope if the current answer is insufficient.

If the current answer is sufficient, or if no further refinement is possible/necessary, return the original query unchanged.

Original Query: {original_query}
Conversation History: {history}
Current Answer: {current_answer}

Refined Query:""",
    input_variables=["original_query", "history", "current_answer"],
)

def refinement_component(state: RAGState) -> RAGState:
    """
    Refines the original query based on the current answer and conversation history using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', 'history', and 'final_answer'.

    Returns:
        Updated RAGState with 'refined_query' (string).
    """
    original_query = state.get('question')
    llm = state.get('llm')
    history = state.get('history', [])
    current_answer = state.get('final_answer', "No answer yet.")

    if not original_query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for refinement.")
        state['refined_query'] = original_query
        return state

    print(f"Attempting to refine query: '{original_query}' based on current answer.")

    # Format history for the prompt
    formatted_history = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history]) if history else "None"

    chain = REFINEMENT_PROMPT | llm | StrOutputParser()

    try:
        refined_query = chain.invoke({
            "original_query": original_query,
            "history": formatted_history,
            "current_answer": current_answer
        }).strip()
        
        state['refined_query'] = refined_query
        print(f"Refined query: {refined_query}")
    except Exception as e:
        print(f"Error during query refinement: {e}. Falling back to original query.")
        state['refined_query'] = original_query
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running refinement_component example...")
    
    # Dummy setup for testing
    import configparser
    
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'LLM': {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'},
        'ollama': {'BASE_URL': 'http://localhost:11434'}
    })

    llm = load_llm(test_config['LLM']['PROVIDER'], model=test_config['LLM']['MODEL'], base_url=test_config['ollama']['BASE_URL'])

    # Test 1: Insufficient answer, needs refinement
    initial_state_insufficient: RAGState = RAGState(
        question="What is the capital of France?",
        history=[],
        llm=llm,
        final_answer="Paris is a city in France.", # Insufficient, doesn't state it's the capital
        config=test_config
    )
    updated_state_insufficient = refinement_component(initial_state_insufficient)
    print(f"\nInsufficient Answer Refinement Result: {updated_state_insufficient.get('refined_query')}")

    # Test 2: Sufficient answer, no refinement needed
    initial_state_sufficient: RAGState = RAGState(
        question="What is the capital of France?",
        history=[],
        llm=llm,
        final_answer="The capital of France is Paris.",
        config=test_config
    )
    updated_state_sufficient = refinement_component(initial_state_sufficient)
    print(f"Sufficient Answer Refinement Result: {updated_state_sufficient.get('refined_query')}")

    # Test 3: With history
    initial_state_history: RAGState = RAGState(
        question="Tell me more about its history.",
        history=[{"query": "What is the capital of France?", "answer": "Paris."}],
        llm=llm,
        final_answer="Paris has a long history.",
        config=test_config
    )
    updated_state_history = refinement_component(initial_state_history)
    print(f"History Refinement Result: {updated_state_history.get('refined_query')}")
