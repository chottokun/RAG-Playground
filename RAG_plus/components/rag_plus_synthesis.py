import sys
import os
from typing import List, Any

# Add the project root to sys.path to allow importing from shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from RAG_plus.shared_types import RAGState
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def rag_plus_synthesis_component(state: RAGState) -> RAGState:
    """
    Generates a final answer based on retrieved knowledge and application examples using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm',
               'knowledge_documents', and 'application_documents'.

    Returns:
        Updated RAGState with 'final_answer' containing the generated response.
    """
    question = state.get('question')
    llm = state.get('llm')
    knowledge_documents: List[Document] = state.get('knowledge_documents', [])
    application_documents: List[Document] = state.get('application_documents', [])

    if not question or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for synthesis.")
        state['final_answer'] = "Error: Missing question or LLM for synthesis."
        return state

    if not knowledge_documents and not application_documents:
        print("Warning: No knowledge or application documents provided for synthesis. Generating answer based on question only.")
        knowledge_context = "No relevant knowledge found."
        application_context = "No relevant application examples found."
    else:
        knowledge_context = "\n---\n".join([doc.page_content for doc in knowledge_documents])
        application_context = "\n---\n".join([doc.page_content for doc in application_documents])
    
    # Prompt template as described in docmuments/Application-Aware_Reasoning.md
    prompt_template = PromptTemplate.from_template(
        """
        Based on the following knowledge and its application examples, answer the user's question.

        ## Reference Knowledge:
        {knowledge}

        ## Application Example of the Knowledge:
        {application}

        ## User Question:
        {question}

        Answer:
        """
    )

    chain = prompt_template | llm | StrOutputParser()

    print("Generating final answer using RAG+ synthesis...")
    try:
        final_answer = chain.invoke({
            "knowledge": knowledge_context,
            "application": application_context,
            "question": question
        })
        state['final_answer'] = final_answer
        print("Final answer generated.")
    except Exception as e:
        print(f"Error during RAG+ synthesis: {e}")
        state['final_answer'] = f"Error generating answer: {e}"
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running rag_plus_synthesis_component example...")
    
    # Dummy setup for testing
    from model_loader.load_llm import load_llm
    import configparser
    from typing import List

    # Create a dummy config
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'LLM': {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'},
        'ollama': {'BASE_URL': 'http://localhost:11434'}
    })

    llm = load_llm(test_config['LLM']['PROVIDER'], model=test_config['LLM']['MODEL'], base_url=test_config['ollama']['BASE_URL'])

    # Create dummy knowledge and application documents
    dummy_knowledge_docs = [
        Document(page_content="Definition: A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.", metadata={"type": "knowledge", "id": "k_001"}),
    ]
    dummy_application_docs = [
        Document(page_content="Example: The numbers 2, 3, 5, 7, 11 are prime numbers.", metadata={"type": "application", "knowledge_id": "k_001"}),
        Document(page_content="Problem: Is 97 a prime number? Solution: Yes, it is.", metadata={"type": "application", "knowledge_id": "k_001"}),
    ]

    # Create a dummy RAGState
    initial_state: RAGState = RAGState(
        question="What is a prime number and give an example?",
        history=[],
        llm=llm,
        knowledge_documents=dummy_knowledge_docs,
        application_documents=dummy_application_docs,
        config=test_config # Not directly used by this component, but part of state
    )

    # Run the component
    updated_state = rag_plus_synthesis_component(initial_state)

    # Check results
    final_answer = updated_state.get('final_answer')
    if final_answer:
        print("\nGenerated Final Answer:")
        print(final_answer)
    else:
        print("\nNo final answer generated.")
