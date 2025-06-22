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

# Prompt template for query decomposition
QUERY_DECOMPOSITION_PROMPT = PromptTemplate(
    template="""Given the following user query and conversation history, decompose the query into a set of simpler, atomic sub-queries that can be answered independently. If the query is already simple, return the original query as a single sub-query.

Each sub-query should be a concise, standalone question.

User Query: {query}
Conversation History: {history}

Provide the sub-queries as a comma-separated list. Do not include any other text or explanations.

Example:
User Query: What is the capital of France and what is its population?
Sub-queries: What is the capital of France?, What is the population of Paris?

User Query: Tell me about the history of AI.
Sub-queries: What is the history of AI?

Sub-queries:""",
    input_variables=["query", "history"],
)

def query_decomposition_component(state: RAGState) -> RAGState:
    """
    Decomposes a complex user query into simpler sub-queries using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', and 'history'.

    Returns:
        Updated RAGState with 'sub_queries' (list of strings) and 'decomposed_query_text' (original or decomposed).
    """
    query = state.get('question')
    llm = state.get('llm')
    history = state.get('history', [])

    if not query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for query decomposition.")
        state['sub_queries'] = [query]
        state['decomposed_query_text'] = query
        return state

    print(f"Performing query decomposition for query: '{query}'")

    # Format history for the prompt
    formatted_history = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history]) if history else "None"

    chain = QUERY_DECOMPOSITION_PROMPT | llm | StrOutputParser()

    try:
        decomposed_query_str = chain.invoke({"query": query, "history": formatted_history}).strip()
        sub_queries = [q.strip() for q in decomposed_query_str.split(',') if q.strip()]
        
        if not sub_queries: # Fallback if LLM returns empty or malformed
            sub_queries = [query]

        state['sub_queries'] = sub_queries
        state['decomposed_query_text'] = decomposed_query_str # Store the raw decomposed string
        print(f"Decomposed query into: {sub_queries}")
    except Exception as e:
        print(f"Error during query decomposition: {e}. Falling back to original query.")
        state['sub_queries'] = [query]
        state['decomposed_query_text'] = query
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running query_decomposition_component example...")
    
    # Dummy setup for testing
    import configparser
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'LLM': {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'},
        'ollama': {'BASE_URL': 'http://localhost:11434'}
    })

    llm = load_llm(test_config['LLM']['PROVIDER'], model=test_config['LLM']['MODEL'], base_url=test_config['ollama']['BASE_URL'])

    # Test 1: Simple query
    initial_state_simple: RAGState = RAGState(
        question="What is the capital of Japan?",
        history=[],
        llm=llm,
        config=test_config
    )
    updated_state_simple = query_decomposition_component(initial_state_simple)
    print(f"\nSimple Query Result: {updated_state_simple.get('sub_queries')}")

    # Test 2: Complex query
    initial_state_complex: RAGState = RAGState(
        question="What is the capital of France and what is its population?",
        history=[],
        llm=llm,
        config=test_config
    )
    updated_state_complex = query_decomposition_component(initial_state_complex)
    print(f"Complex Query Result: {updated_state_complex.get('sub_queries')}")

    # Test 3: Query with history
    initial_state_history: RAGState = RAGState(
        question="And what about Germany?",
        history=[{"query": "What is the capital of France?", "answer": "Paris."}],
        llm=llm,
        config=test_config
    )
    updated_state_history = query_decomposition_component(initial_state_history)
    print(f"History Query Result: {updated_state_history.get('sub_queries')}")
