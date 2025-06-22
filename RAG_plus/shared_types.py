import sys
import os
from typing import Dict, Any, List, TypedDict

# Add the project root to sys.path to allow importing from shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust path for RAG_plus
sys.path.append(project_root)

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
