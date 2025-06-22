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

# Prompt template for synthesis
SYNTHESIS_PROMPT = PromptTemplate(
    template="""Given the following documents, synthesize a concise and informative answer to the user's query.

User Query: {query}

Relevant Documents:
{documents}

Synthesized Answer:""",
    input_variables=["query", "documents"],
)

def synthesis_component(state: RAGState, max_tokens: int = 512, temperature: float = 0.7) -> RAGState:
    """
    Synthesizes a final answer based on retrieved documents using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', and 'retrieved_documents'.
        max_tokens: The maximum number of tokens to generate in the answer.
        temperature: The temperature to use for answer generation.

    Returns:
        Updated RAGState with 'final_answer' (string).
    """
    query = state.get('question')
    llm = state.get('llm')
    retrieved_documents: List[Document] = state.get('retrieved_documents', [])

    if not query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for synthesis.")
        state['final_answer'] = "Error: Missing question or LLM for synthesis."
        return state

    if not retrieved_documents:
        print("No documents to synthesize. Returning a default message.")
        state['final_answer'] = "No relevant documents found to answer the query."
        return state

    print(f"Synthesizing answer for query: '{query}' using {len(retrieved_documents)} documents.")

    # Format documents for the prompt
    formatted_documents = "\n---\n".join([f"Document {i+1}: {doc.page_content[:500]}..." for i, doc in enumerate(retrieved_documents)])

    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()

    try:
        final_answer = chain.invoke({
            "query": query,
            "documents": formatted_documents
        }).strip()
        state['final_answer'] = final_answer
        print(f"Synthesized answer: {final_answer[:200]}...")
    except Exception as e:
        print(f"Error during synthesis: {e}")
        state['final_answer'] = f"Error generating final answer: {e}"
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running synthesis_component example...")
    
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
        Document(page_content="Document 1: The capital of France is Paris. It is a major European city.", metadata={"id": "doc_paris_1"}),
        Document(page_content="Document 2: Paris is located on the river Seine.", metadata={"id": "doc_paris_2"}),
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
    updated_state = synthesis_component(initial_state, max_tokens=256, temperature=0.5)

    # Check results
    final_answer = updated_state.get('final_answer')
    if final_answer:
        print(f"\nSynthesized Answer: {final_answer}")
    else:
        print("\nNo final answer generated.")
