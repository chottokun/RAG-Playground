import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from RAG_plus.shared_types import RAGState
from langchain_core.documents import Document

def application_retrieval_component(state: RAGState) -> RAGState:
    """
    Retrieves application examples linked to knowledge documents from the application corpus.

    Args:
        state: The current RAGState, expected to contain 'knowledge_documents' and 'application_vectorstore'.

    Returns:
        Updated RAGState with 'application_documents' containing the retrieved documents.
    """
    knowledge_documents: List[Document] = state.get('knowledge_documents', [])
    application_vectorstore = state.get('application_vectorstore')

    if not knowledge_documents:
        print("No knowledge documents found in RAGState. Skipping application retrieval.")
        state['application_documents'] = []
        return state
    
    if not application_vectorstore:
        print("Error: 'application_vectorstore' not found in RAGState. Cannot perform application retrieval.")
        state['application_documents'] = []
        return state

    all_application_docs: List[Document] = []
    retrieved_knowledge_ids = [doc.metadata.get('id') for doc in knowledge_documents if doc.metadata.get('id')]
    print(f"Retrieving application examples for knowledge IDs: {retrieved_knowledge_ids}")

    for knowledge_id in retrieved_knowledge_ids:
        try:
            # Filter by both knowledge_id and type in the DB query for efficiency
            app_docs = application_vectorstore.similarity_search(
                query=" ",  # Dummy query as filtering is the main goal
                k=5,  # Retrieve up to 5 matching application docs
                filter={
                    "$and": [
                        {"knowledge_id": {"$eq": knowledge_id}},
                        {"type": {"$eq": "application"}},
                    ]
                },
            )
            if app_docs:
                all_application_docs.extend(app_docs)
                print(f"  Found {len(app_docs)} application docs for knowledge ID: {knowledge_id}")
            else:
                print(f"  No application docs found for knowledge ID: {knowledge_id}")
        except Exception as e:
            print(f"Error retrieving application docs for knowledge ID {knowledge_id}: {e}")
    
    state['application_documents'] = all_application_docs
    print(f"Retrieved a total of {len(all_application_docs)} application documents.")
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running application_retrieval_component example...")
    
    # Dummy setup for testing
    import configparser
    from langchain_chroma import Chroma
    from shared_components.model_loader.load_embedding import load_embedding_model
    from typing import List

    # Create a dummy config
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'embedding': {'MODEL': 'intfloat/multilingual-e5-small'},
        'vectorstore': {
            'KNOWLEDGE_DIRECTORY': './test_vectorstore_knowledge',
            'APPLICATION_DIRECTORY': './test_vectorstore_application'
        }
    })

    embedding_model = load_embedding_model(test_config['embedding']['MODEL'])
    
    # Initialize dummy vectorstores
    knowledge_dir = test_config['vectorstore']['KNOWLEDGE_DIRECTORY']
    application_dir = test_config['vectorstore']['APPLICATION_DIRECTORY']
    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(application_dir, exist_ok=True)

    dummy_knowledge_vectorstore = Chroma(persist_directory=knowledge_dir, embedding_function=embedding_model)
    dummy_application_vectorstore = Chroma(persist_directory=application_dir, embedding_function=embedding_model)

    # Add some dummy knowledge documents
    dummy_knowledge_docs = [
        Document(page_content="Definition: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.", metadata={"type": "knowledge", "id": "k_001"}),
        Document(page_content="Theorem: The Pythagorean theorem states that in a right-angled triangle, the square of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the other two sides.", metadata={"type": "knowledge", "id": "k_002"}),
    ]
    dummy_knowledge_vectorstore.add_documents(dummy_knowledge_docs)

    # Add some dummy application documents linked to knowledge
    dummy_application_docs = [
        Document(page_content="Example: The numbers 2, 3, 5, 7, 11 are prime numbers.", metadata={"type": "application", "knowledge_id": "k_001"}),
        Document(page_content="Problem: Is 97 a prime number? Solution: Yes, it is.", metadata={"type": "application", "knowledge_id": "k_001"}),
        Document(page_content="Application: If a triangle has sides 3, 4, and 5, is it a right-angled triangle? Solution: Yes, because 3^2 + 4^2 = 9 + 16 = 25 = 5^2.", metadata={"type": "application", "knowledge_id": "k_002"}),
    ]
    dummy_application_vectorstore.add_documents(dummy_application_docs)

    print(f"Dummy knowledge vectorstore created with {dummy_knowledge_vectorstore._collection.count()} documents.")
    print(f"Dummy application vectorstore created with {dummy_application_vectorstore._collection.count()} documents.")

    # Create a dummy RAGState with retrieved knowledge documents
    initial_state: RAGState = RAGState(
        question="How to apply prime numbers?",
        history=[],
        knowledge_documents=[
            Document(page_content="Definition: A prime number...", metadata={"type": "knowledge", "id": "k_001"}),
            Document(page_content="Theorem: Pythagorean theorem...", metadata={"type": "knowledge", "id": "k_002"}),
        ],
        application_vectorstore=dummy_application_vectorstore,
        config=test_config,
        llm=None # Not needed for this component
    )

    # Run the component
    updated_state = application_retrieval_component(initial_state)

    # Check results
    retrieved_app_docs = updated_state.get('application_documents')
    if retrieved_app_docs:
        print("\nRetrieved Application Documents:")
        for doc in retrieved_app_docs:
            print(f"- Content: {doc.page_content[:100]}...")
            print(f"  Metadata: {doc.metadata}")
    else:
        print("\nNo application documents retrieved.")

    # Clean up test directories
    # import shutil
    # if os.path.exists(knowledge_dir):
    #     shutil.rmtree(knowledge_dir)
    # if os.path.exists(application_dir):
    #     shutil.rmtree(application_dir)
