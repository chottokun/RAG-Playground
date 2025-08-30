import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from shared_components.model_loader and shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from shared_components.model_loader.load_llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

# Import shared types
from RAG_plus.shared_types import RAGState
from typing import TypedDict

# Define the expected output structure for the LLM evaluation of retrieved documents
class DocumentEvaluationResult(TypedDict):
    document_id: str
    relevance_score: float
    reasoning: str

# Prompt template for the LLM evaluator for documents
DOCUMENT_EVALUATOR_PROMPT = PromptTemplate(
    template="""You are an expert document evaluator. Given a user query and a document, assess the relevance of the document to the query.

Provide your evaluation in JSON format, including the 'document_id', a 'relevance_score' (0.0 to 1.0), and 'reasoning' for your score.

User Query: {query}
Document: {document_content}

Output JSON:
Strictly follow the above format and output only JSON. Do not include any explanations or extra text.
""",
    input_variables=["query", "document_content"],
)

def evaluation_component(state: RAGState) -> RAGState:
    """
    Evaluates the relevance of retrieved documents to the original query using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', and 'retrieved_documents'.

    Returns:
        Updated RAGState with 'evaluated_documents' (list of DocumentEvaluationResult).
    """
    query = state.get('question')
    llm = state.get('llm')
    retrieved_documents: List[Document] = state.get('retrieved_documents', [])

    if not query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for evaluation.")
        state['evaluated_documents'] = []
        return state

    if not retrieved_documents:
        print("No documents to evaluate. Skipping evaluation component.")
        state['evaluated_documents'] = []
        return state

    print(f"Evaluating {len(retrieved_documents)} documents for query: '{query}'")

    parser = JsonOutputParser()
    chain = DOCUMENT_EVALUATOR_PROMPT | llm | parser

    evaluated_docs: List[DocumentEvaluationResult] = []
    for i, doc in enumerate(retrieved_documents):
        print(f"  Evaluating document {i+1}/{len(retrieved_documents)} (ID: {doc.metadata.get('id', 'N/A')})...")
        try:
            evaluation_result = chain.invoke({
                "query": query,
                "document_content": doc.page_content
            })
            # Basic validation and casting
            if isinstance(evaluation_result, dict) and 'relevance_score' in evaluation_result and 'document_id' in evaluation_result:
                evaluated_docs.append(DocumentEvaluationResult(
                    document_id=str(doc.metadata.get('id', f"doc_{i}")), # Use actual ID or generated
                    relevance_score=float(evaluation_result['relevance_score']),
                    reasoning=str(evaluation_result.get('reasoning', ''))
                ))
            else:
                print(f"    Warning: LLM evaluation output for document {i} did not match expected structure. Skipping.")
        except Exception as e:
            print(f"    Error evaluating document {i} (ID: {doc.metadata.get('id', 'N/A')}): {e}. Skipping.")
    
    state['evaluated_documents'] = evaluated_docs
    print(f"Finished evaluating documents. Total evaluated: {len(evaluated_docs)}")
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running evaluation_component example...")
    
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
        Document(page_content="The capital of France is Paris. It is known for the Eiffel Tower.", metadata={"id": "doc_paris"}),
        Document(page_content="The capital of Germany is Berlin. It has a rich history.", metadata={"id": "doc_berlin"}),
        Document(page_content="The Amazon rainforest is a vast tropical forest in South America.", metadata={"id": "doc_amazon"}),
    ]

    # Create a dummy RAGState
    initial_state: RAGState = RAGState(
        question="What is the capital of France?",
        history=[],
        llm=llm,
        retrieved_documents=dummy_retrieved_docs,
        config=test_config
    )

    # Run the component
    updated_state = evaluation_component(initial_state)

    # Check results
    evaluated_docs = updated_state.get('evaluated_documents')
    if evaluated_docs:
        print("\nEvaluated Documents:")
        for doc_eval in evaluated_docs:
            print(f"- Doc ID: {doc_eval['document_id']}, Score: {doc_eval['relevance_score']:.2f}, Reasoning: {doc_eval['reasoning']}")
    else:
        print("\nNo documents evaluated.")
