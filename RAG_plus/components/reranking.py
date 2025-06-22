import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from model_loader and shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from model_loader.load_llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Import shared types
from RAG_plus.shared_types import RAGState

# Prompt template for the LLM reranker
RERANKING_PROMPT = PromptTemplate(
    template="""You are an expert document reranker. Given a user query and a list of documents, re-order the documents based on their relevance to the query.

Provide the re-ordered document IDs as a comma-separated list. Do not include any other text or explanations.

User Query: {query}

Documents:
{documents}

Re-ordered Document IDs:""",
    input_variables=["query", "documents"],
)

def reranking_component(state: RAGState, method: str = "llm", top_k: int = 5) -> RAGState:
    """
    Re-orders the retrieved documents based on their relevance to the query using an LLM or other method.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', and 'retrieved_documents'.
        method: The reranking method to use ("llm" or other).
        top_k: The number of top documents to keep after reranking.

    Returns:
        Updated RAGState with 'reranked_documents' (list of Document).
    """
    query = state.get('question')
    llm = state.get('llm')
    retrieved_documents: List[Document] = state.get('retrieved_documents', [])

    if not query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for reranking.")
        state['reranked_documents'] = retrieved_documents[:top_k] # Keep original order
        return state

    if not retrieved_documents:
        print("No documents to rerank. Skipping reranking component.")
        state['reranked_documents'] = retrieved_documents[:top_k] # Keep original order
        return state

    print(f"Reranking {len(retrieved_documents)} documents for query: '{query}' using method '{method}'")

    if method == "llm":
        try:
            # Format documents for the prompt
            formatted_documents = "\n---\n".join([f"Document {i+1} (ID: {doc.metadata.get('id', 'N/A')}): {doc.page_content[:200]}..." for i, doc in enumerate(retrieved_documents)])

            chain = RERANKING_PROMPT | llm | StrOutputParser()
            reranked_ids_str = chain.invoke({
                "query": query,
                "documents": formatted_documents
            }).strip()

            # Parse the re-ordered document IDs
            reranked_ids = [s.strip() for s in reranked_ids_str.split(',') if s.strip()]

            # Map the re-ordered IDs back to the original documents, preserving order
            reranked_documents = []
            for doc_id in reranked_ids:
                for doc in retrieved_documents:
                    if doc.metadata.get('id') == doc_id:
                        reranked_documents.append(doc)
                        break # Found the doc, move to next ID
            
            # If some IDs are missing or invalid, fill with remaining documents
            remaining_docs = [doc for doc in retrieved_documents if doc not in reranked_documents]
            reranked_documents.extend(remaining_docs)

            state['reranked_documents'] = reranked_documents[:top_k] # Truncate to top_k
            print(f"Reranked documents: {[doc.metadata.get('id', 'N/A') for doc in state['reranked_documents']]}")

        except Exception as e:
            print(f"Error during LLM reranking: {e}. Keeping original order.")
            state['reranked_documents'] = retrieved_documents[:top_k] # Keep original order
    else:
        print(f"Unknown reranking method: {method}. Keeping original order.")
        state['reranked_documents'] = retrieved_documents[:top_k] # Keep original order
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running reranking_component example...")
    
    # Dummy setup for testing
    import configparser
    from langchain_core.documents import Document
    
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'LLM': {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'},
        'ollama': {'BASE_URL': 'http://localhost:11434'}
    })

    llm = load_llm(test_config['LLM']['PROVIDER'], model=test_config['LLM']['MODEL'], base_url=test_config['ollama']['BASE_URL'])

    # Create dummy retrieved documents
    dummy_retrieved_docs = [
        Document(page_content="Document 1: The capital of France is Paris.", metadata={"id": "doc_paris"}),
        Document(page_content="Document 2: The Eiffel Tower is a famous landmark in Paris.", metadata={"id": "doc_eiffel"}),
        Document(page_content="Document 3: Berlin is the capital of Germany.", metadata={"id": "doc_berlin"}),
    ]

    # Create a dummy RAGState
    initial_state: RAGState = RAGState(
        question="What is the capital of France?",
        history=[],
        llm=llm,
        retrieved_documents=dummy_retrieved_docs,
        config=test_config
    )

    # Run the component with LLM reranking
    updated_state = reranking_component(initial_state, method="llm", top_k=2)

    # Check results
    reranked_docs = updated_state.get('reranked_documents')
    if reranked_docs:
        print("\nReranked Documents:")
        for doc in reranked_docs:
            print(f"- ID: {doc.metadata.get('id')}, Content: {doc.page_content[:50]}...")
    else:
        print("\nNo documents reranked.")
