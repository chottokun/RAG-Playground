import sys
import os
from typing import Dict, Any, List, TYPE_CHECKING

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting

from langchain_core.documents import Document
# Assuming vectorstore is a Chroma instance or similar with similarity_search

def retrieval_component(state: RAGState, k: int = 5) -> Dict[str, Any]:
    """
    Retrieves documents from the vector store based on the query or sub-queries.

    Args:
        state: The current RAGState, must contain 'question' or 'sub_queries', and 'vectorstore'.
        k: The number of documents to retrieve per query.

    Returns:
        A dictionary to update the state, containing 'retrieved_documents'.
    """
    question = state.get('question')
    sub_queries = state.get('sub_queries') # Use sub_queries if available
    vectorstore = state.get('vectorstore')
    retrieved_documents: List[Document] = []

    if vectorstore is None:
        print("Error: Vectorstore not available in state for retrieval.")
        return {"retrieved_documents": []}

    queries_to_use = []
    if sub_queries and isinstance(sub_queries, list) and len(sub_queries) > 0:
        queries_to_use = sub_queries
        print(f"Retrieving documents for {len(sub_queries)} sub-queries...")
    elif question:
        queries_to_use = [question]
        print(f"Retrieving documents for the main question...")
    else:
        print("Error: No question or sub-queries available in state for retrieval.")
        return {"retrieved_documents": []}

    for query in queries_to_use:
        print(f"Searching for: '{query}' (k={k})")
        try:
            # Perform retrieval for each query
            docs = vectorstore.similarity_search(query, k=k)
            retrieved_documents.extend(docs)
            print(f"Found {len(docs)} documents for '{query}'.")
        except Exception as e:
            print(f"Error during retrieval for query '{query}': {e}")

    # Optional: Deduplicate retrieved_documents if needed
    # For simplicity, just returning the combined list for now
    print(f"Total retrieved documents: {len(retrieved_documents)}")
    return {"retrieved_documents": retrieved_documents}

# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example retrieval...")
    # This component requires a loaded vectorstore and a question/sub-queries in the state
    print("Vectorstore and queries not loaded for example. Skipping retrieval test.")
