from __future__ import annotations
import sys
import os
from typing import Dict, Any, List, TYPE_CHECKING
import ast # Import ast for safer parsing

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough # Assuming LLM returns string

# Prompt template for query decomposition
DECOMPOSITION_PROMPT = PromptTemplate(
    template="""Given the following question, break it down into a list of simpler sub-queries. If the question is simple, just return the original question as a single item in the list.
Question: {question}
Sub-queries:""",
    input_variables=["question"],
)

def query_decomposition_component(state: RAGState, enable_dynamic_decomposition: bool = True) -> Dict[str, Any]:
    """
    Decomposes a complex query into simpler sub-queries using an LLM.

    Args:
        state: The current RAGState, must contain 'question' and 'llm'.
        enable_dynamic_decomposition: Whether to dynamically decide if decomposition is needed.

    Returns:
        A dictionary to update the state, containing 'sub_queries'.
    """
    question = state['question']
    llm = state['llm']
    sub_queries = []

    print(f"Query Decomposition: Dynamic decomposition enabled = {enable_dynamic_decomposition}")

    if enable_dynamic_decomposition:
        # Implement logic to decide if decomposition is needed (similar to DeepStepRAG)
        # This could involve checking query length, complexity keywords, etc.
        # For demonstration, decompose if the question is long or contains "and", "or"
        if len(question.split()) > 10 or any(word in question.lower() for word in [" and ", " or "]):
            print(f"Query '{question}' is complex. Attempting decomposition.")
            try:
                # Use LLM to decompose the query
                # Assuming LLM returns a string like "['query1', 'query2']" or just "query"
                raw_output = (DECOMPOSITION_PROMPT | llm | RunnablePassthrough()).invoke({"question": question})

                # Robust parsing attempt
                if isinstance(raw_output, str):
                    raw_output = raw_output.strip()
                    if raw_output.startswith('[') and raw_output.endswith(']'):
                         try:
                             # Use ast.literal_eval for safer parsing
                             parsed_output = ast.literal_eval(raw_output)
                             if isinstance(parsed_output, list) and all(isinstance(q, str) for q in parsed_output):
                                 sub_queries = parsed_output
                             else:
                                 print("Warning: LLM output parsed but not a list of strings. Using original query.")
                                 sub_queries = [question] # Fallback
                         except:
                             print("Warning: LLM output could not be parsed as a list. Using original query.")
                             sub_queries = [question] # Fallback
                    else:
                         sub_queries = [raw_output] # Treat as single query
                elif isinstance(raw_output, list) and all(isinstance(q, str) for q in raw_output):
                     sub_queries = raw_output
                else:
                     print("Warning: LLM returned unexpected type. Using original query.")
                     sub_queries = [question] # Fallback

                print("Decomposed Sub-queries:", sub_queries)

            except Exception as e:
                print(f"Error during query decomposition LLM call: {e}. Proceeding with original query.")
                sub_queries = [question]
        else:
            print("Query is simple. No decomposition needed.")
            sub_queries = [question]
    else:
        print("Dynamic decomposition is disabled. Proceeding with original query.")
        sub_queries = [question]

    return {"sub_queries": sub_queries}

# Example Usage (for testing)
if __name__ == "__main__":
    # This part would typically be run from the orchestrator
    print("Running example query decomposition...")
    # Assuming you have a way to load your LLM here for testing
    # from model_loader.load_llm import load_llm
    # llm_provider = "ollama" # Or read from config
    # llm_model = "gemma3:4b-it-qat" # Or read from config
    # llm_base_url = "http://localhost:11434" # Or read from config
    # test_llm = load_llm(llm_provider, model=llm_model, base_url=llm_base_url)

    # if 'test_llm' in locals():
    #     # Create a dummy state
    #     dummy_state: RAGState = RAGState(question="What are the environmental impacts of electric vehicles and how do they compare to gasoline cars?", history=[])
    #     dummy_state['llm'] = test_llm

    #     result_state_update = query_decomposition_component(dummy_state)
    #     print("Resulting state update:", result_state_update)

    #     dummy_state_simple: RAGState = RAGState(question="What is a RAG system?", history=[])
    #     dummy_state_simple['llm'] = test_llm
    #     result_state_update_simple = query_decomposition_component(dummy_state_simple)
    #     print("Resulting state update (simple query):", result_state_update_simple)

    # else:
    print("LLM not loaded for example. Skipping query decomposition test.")
