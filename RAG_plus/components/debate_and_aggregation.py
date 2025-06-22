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

# Define prompt templates for debate and aggregation
DEBATE_PROMPT = PromptTemplate(
    template="""You are a debater tasked with examining the following documents and answering the user's query. Provide a well-reasoned argument based on the information in the documents. Be concise and persuasive.

User Query: {query}

Relevant Documents:
{documents}

Your Argument:""",
    input_variables=["query", "documents"],
)

AGGREGATION_PROMPT = PromptTemplate(
    template="""You are an expert at synthesizing information from multiple sources. Given the following arguments from different debaters, synthesize a final answer that incorporates the key points from each argument.

Arguments:
{arguments}

Final Answer:""",
    input_variables=["arguments"],
)

def debate_and_aggregation_component(state: RAGState, num_agents: int = 3, max_rounds: int = 2) -> RAGState:
    """
    Conducts a multi-agent debate to synthesize information from documents and aggregates the results.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', and 'retrieved_documents'.
        num_agents: The number of debating agents.
        max_rounds: The maximum number of debate rounds.

    Returns:
        Updated RAGState with 'debate_outputs' (list of strings) and 'final_answer' (aggregated answer).
    """
    query = state.get('question')
    llm = state.get('llm')
    retrieved_documents: List[Document] = state.get('retrieved_documents', [])

    if not query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for debate.")
        state['debate_outputs'] = []
        state['final_answer'] = "Error: Missing question or LLM for debate."
        return state

    if not retrieved_documents:
        print("No documents to debate. Skipping debate component.")
        state['debate_outputs'] = []
        state['final_answer'] = "No documents available for debate."
        return state

    print(f"Conducting multi-agent debate with {num_agents} agents and {max_rounds} rounds for query: '{query}'")

    # --- Debate Phase ---
    debate_outputs = []
    for i in range(num_agents):
        print(f"  Agent {i+1}/{num_agents}: Generating argument...")
        try:
            # Select a subset of documents for each agent (simple distribution)
            agent_documents = retrieved_documents[i % len(retrieved_documents):] # Ensure at least one doc
            
            # Format documents for the prompt
            formatted_documents = "\n---\n".join([f"Document {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(agent_documents)])

            chain = DEBATE_PROMPT | llm | StrOutputParser()
            agent_argument = chain.invoke({
                "query": query,
                "documents": formatted_documents
            }).strip()
            debate_outputs.append(agent_argument)
            print(f"    Agent {i+1} argument: {agent_argument[:100]}...")
        except Exception as e:
            print(f"    Error generating argument for agent {i+1}: {e}")
            debate_outputs.append(f"Error generating argument: {e}")

    state['debate_outputs'] = debate_outputs

    # --- Aggregation Phase ---
    print("Synthesizing final answer from debate arguments...")
    try:
        formatted_arguments = "\n---\n".join([f"Agent {i+1}: {arg[:300]}..." for i, arg in enumerate(debate_outputs)])
        chain = AGGREGATION_PROMPT | llm | StrOutputParser()
        final_answer = chain.invoke({
            "arguments": formatted_arguments
        }).strip()
        state['final_answer'] = final_answer
        print(f"Final answer: {final_answer[:200]}...")
    except Exception as e:
        print(f"Error during aggregation: {e}")
        state['final_answer'] = f"Error generating final answer: {e}"
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running debate_and_aggregation_component example...")
    
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
        Document(page_content="Document 1: The capital of France is Paris. It is a major European city and a global center for art, fashion, gastronomy and culture.", metadata={"id": "doc_paris_1"}),
        Document(page_content="Document 2: Paris is located on the river Seine, in northern France, at the heart of the Île-de-France region.", metadata={"id": "doc_paris_2"}),
        Document(page_content="Document 3: The city is also known for its museums and architectural landmarks.", metadata={"id": "doc_paris_3"}),
    ]

    # Create a dummy RAGState
    initial_state: RAGState = RAGState(
        question="What is Paris known for?",
        history=[],
        llm=llm,
        retrieved_documents=dummy_retrieved_docs,
        config=test_config
    )

    # Run the component
    updated_state = debate_and_aggregation_component(initial_state, num_agents=2, max_rounds=1)

    # Check results
    debate_outputs = updated_state.get('debate_outputs')
    final_answer = updated_state.get('final_answer')

    if debate_outputs:
        print("\nDebate Outputs:")
        for i, arg in enumerate(debate_outputs):
            print(f"- Agent {i+1}: {arg[:150]}...")
    else:
        print("\nNo debate outputs generated.")

    if final_answer:
        print(f"\nFinal Answer: {final_answer}")
    else:
        print("\nNo final answer generated.")
