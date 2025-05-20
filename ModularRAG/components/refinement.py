import sys
import os
from typing import Dict, Any, List

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.prompts import PromptTemplate

# Prompt template for query refinement
REFINEMENT_PROMPT = PromptTemplate(
    template="""Given the original question and feedback on the relevance of retrieved documents, generate a refined query that would lead to better search results.
Original Question: {question}
Document Evaluation Feedback:
{eval_context}
Refined Query:""",
    input_variables=["question", "eval_context"],
)

def refinement_component(state: RAGState) -> Dict[str, Any]:
    """
    Generates a refined query based on evaluation results using an LLM.

    Args:
        state: The current RAGState, must contain 'question', 'evaluation_results', and 'llm'.

    Returns:
        A dictionary to update the state, containing 'refined_query'.
    """
    question = state.get('question')
    evaluation_results = state.get('evaluation_results', [])
    llm = state.get('llm')
    refined_query = question # Default to original question

    if question is None or llm is None:
        print("Error: Question or LLM not available in state for refinement.")
        return {"refined_query": refined_query}

    if not evaluation_results:
        print("No evaluation results to refine the query. Using original question.")
        return {"refined_query": refined_query}

    print("Refining query based on evaluation results...")

    # Prepare context for refinement prompt
    eval_context = "\n".join([
        f"Document {i+1} (Score: {res['evaluation'].get('relevance_score', 'N/A')}): {res['evaluation'].get('reason', 'N/A')}"
        for i, res in enumerate(evaluation_results)
    ])

    refine_chain = REFINEMENT_PROMPT | llm # Assuming LLM returns a string for the refined query

    try:
        refined_query = refine_chain.invoke({"question": question, "eval_context": eval_context})
        print("Refined Query:", refined_query)
    except Exception as e:
        print(f"Error during query refinement LLM call: {e}. Using original question as refined query.")
        refined_query = question

    return {"refined_query": refined_query}

# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example refinement...")
    # This component requires a loaded LLM and evaluation results in the state
    print("LLM and evaluation results not loaded for example. Skipping refinement test.")
