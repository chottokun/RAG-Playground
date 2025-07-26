import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from RAG_plus.shared_types import RAGState
from langchain_core.documents import Document

def knowledge_retrieval_component(state: RAGState, k: int = 5) -> RAGState:
    """
    Retrieves relevant knowledge documents from the knowledge corpus based on the query.

    Args:
        state: The current RAGState, expected to contain 'question' and 'knowledge_vectorstore'.
        k: The number of top knowledge documents to retrieve.

    Returns:
        Updated RAGState with 'knowledge_documents' containing the retrieved documents.
    """
    question = state.get('question')
    knowledge_vectorstore = state.get('knowledge_vectorstore')

    if not question:
        print("Error: 'question' not found in RAGState for knowledge retrieval.")
        state['knowledge_documents'] = []
        return state
    
    if not knowledge_vectorstore:
        print("Error: 'knowledge_vectorstore' not found in RAGState. Cannot perform knowledge retrieval.")
        state['knowledge_documents'] = []
        return state

    print(f"Performing knowledge retrieval for query: '{question}' with k={k}")
    try:
        # Filter for documents of type 'knowledge'
        retriever = knowledge_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": {"type": "knowledge"}}
        )
        knowledge_documents: List[Document] = retriever.invoke(question)
        state['knowledge_documents'] = knowledge_documents
        print(f"Retrieved {len(knowledge_documents)} knowledge documents.")
    except Exception as e:
        print(f"Error during knowledge retrieval: {e}")
        state['knowledge_documents'] = []
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running knowledge_retrieval_component example...")
    
    # Dummy setup for testing
    import configparser
    from langchain_chroma import Chroma
    from shared_components.model_loader.load_embedding import load_embedding_model
    from typing import List

    # Create a dummy config
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'embedding': {'MODEL': 'intfloat/multilingual-e5-small'},
        'vectorstore': {'KNOWLEDGE_DIRECTORY': './test_vectorstore_knowledge'}
    })

    embedding_model = load_embedding_model(test_config['embedding']['MODEL'])
    
    # Initialize a dummy knowledge vectorstore
    knowledge_dir = test_config['vectorstore']['KNOWLEDGE_DIRECTORY']
    os.makedirs(knowledge_dir, exist_ok=True)
    dummy_knowledge_vectorstore = Chroma(persist_directory=knowledge_dir, embedding_function=embedding_model)

    # Add some dummy knowledge documents
    dummy_knowledge_docs = [
        Document(page_content="Definition: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.", metadata={"type": "knowledge", "id": "k_001"}),
        Document(page_content="Theorem: The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides.", metadata={"type": "knowledge", "id": "k_002"}),
        Document(page_content="This is an unrelated application example.", metadata={"type": "application", "knowledge_id": "k_001"}),
    ]
    dummy_knowledge_vectorstore.add_documents(dummy_knowledge_docs)
    print(f"Dummy knowledge vectorstore created with {dummy_knowledge_vectorstore._collection.count()} documents.")

    # Create a dummy RAGState
    initial_state: RAGState = RAGState(
        question="What is a prime number?",
        history=[],
        knowledge_vectorstore=dummy_knowledge_vectorstore,
        config=test_config,
        llm=None # Not needed for this component, but part of RAGState
    )

    # Run the component
    updated_state = knowledge_retrieval_component(initial_state, k=1)

    # Check results
    retrieved_docs = updated_state.get('knowledge_documents')
    if retrieved_docs:
        print("\nRetrieved Knowledge Documents:")
        for doc in retrieved_docs:
            print(f"- Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
    else:
        print("\nNo knowledge documents retrieved.")

    # Clean up test directories
    # import shutil
    # if os.path.exists(knowledge_dir):
    #     shutil.rmtree(knowledge_dir)
