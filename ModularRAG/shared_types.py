from typing import Dict, Any, List, TypedDict

# --- State Management ---
# A simple dictionary to hold the state as components execute
class RAGState(Dict[str, Any]):
    """Represents the state passed between RAG components."""
    question: str
    history: List['HistoryItem'] # Use forward reference for HistoryItem
    # Add other potential state variables as needed by components
    # e.g., retrieved_documents, evaluation_results, refined_query, debate_outputs, final_answer

# Define the structure for history items
class HistoryItem(TypedDict):
    query: str
    answer: str
