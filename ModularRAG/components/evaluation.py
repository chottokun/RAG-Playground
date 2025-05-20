import sys
import os
from typing import Dict, Any, List

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser # Assuming JSON output for evaluation

# Prompt template for document evaluation
EVALUATION_PROMPT = PromptTemplate(
    template="""You are a Relevance Evaluation Agent. Given an original question and a document, rate the document's relevance to the question on a scale of 1 to 5 (1 being not relevant, 5 being highly relevant). Also, provide a brief reason for your rating.
Original Question: {question}
Document: {document_content}
Output JSON: {{'relevance_score': int, 'reason': str}}""",
    input_variables=["question", "document_content"],
)

def evaluation_component(state: RAGState) -> Dict[str, Any]:
    """
    Evaluates the relevance of retrieved documents to the original question using an LLM.

    Args:
        state: The current RAGState, must contain 'question', 'retrieved_documents', and 'llm'.

    Returns:
        A dictionary to update the state, containing 'evaluation_results'.
    """
    question = state.get('question')
    retrieved_documents = state.get('retrieved_documents', [])
    llm = state.get('llm')
    evaluation_results: List[Dict[str, Any]] = []

    if question is None or llm is None:
        print("Error: Question or LLM not available in state for evaluation.")
        return {"evaluation_results": []}

    if not retrieved_documents:
        print("No documents to evaluate.")
        return {"evaluation_results": []}

    print(f"Evaluating {len(retrieved_documents)} documents...")

    parser = JsonOutputParser()
    eval_chain = EVALUATION_PROMPT | llm | parser

    for i, doc in enumerate(retrieved_documents):
        print(f"Evaluating document {i+1}/{len(retrieved_documents)}")
        try:
            result = eval_chain.invoke({"question": question, "document_content": doc.page_content})
            # Basic validation of the output structure
            if not isinstance(result, dict) or 'relevance_score' not in result or 'reason' not in result:
                 print(f"Warning: LLM evaluation output for document {i+1} did not match expected structure.")
                 result = {"relevance_score": 1, "reason": "LLM output format incorrect."} # Fallback

            evaluation_results.append({"document": doc, "evaluation": result})
            print(f"  Doc {i+1} Score: {result.get('relevance_score', 'N/A')}, Reason: {result.get('reason', 'N/A')[:100]}...") # Show snippet
        except Exception as e:
            print(f"Error during evaluation for document {i+1}: {e}")
            evaluation_results.append({"document": doc, "evaluation": {"relevance_score": 1, "reason": f"Evaluation failed: {e}"}}) # Default to low score on failure

    print(f"Completed evaluation for {len(evaluation_results)} documents.")
    return {"evaluation_results": evaluation_results}

# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example evaluation...")
    # This component requires a loaded LLM and retrieved documents in the state
    print("LLM and documents not loaded for example. Skipping evaluation test.")
