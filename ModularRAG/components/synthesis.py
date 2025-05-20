import sys
import os
from typing import Dict, Any, List

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Prompt template for final answer synthesis (can be adapted)
SYNTHESIS_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
Answer concisely.
"""
)

def synthesis_component(state: RAGState) -> Dict[str, Any]:
    """
    Generates the final answer based on the provided context using an LLM.

    Args:
        state: The current RAGState, must contain 'question', 'llm', and
               either 'reranked_documents', 'debate_results' (for aggregated summary),
               or other relevant context in the state.

    Returns:
        A dictionary to update the state, containing 'final_answer'.
    """
    question = state.get('question')
    llm = state.get('llm')
    final_answer = "Could not generate answer."

    if question is None or llm is None:
        print("Error: Question or LLM not available in state for synthesis.")
        return {"final_answer": final_answer}

    # Determine the context to use for synthesis
    # Prioritize aggregated debate results if available, otherwise use reranked documents
    context = ""
    if 'debate_results' in state and state['debate_results']:
        # Use the final aggregated summary from the debate component
        # Assuming the debate component stores the final summary in the state
        # Need to check the structure of debate_results from debate_and_aggregation.py
        # It returns a list of round traces, each with 'agg_summary'.
        # Let's use the last round's agg_summary as the primary context.
        last_round_summary = state['debate_results'][-1].get('agg_summary') if state['debate_results'] else None
        if last_round_summary and last_round_summary != "None": # Check against the default "None" string
             context = f"Aggregated Summary:\n{last_round_summary}"
             print("Using aggregated summary for synthesis.")
        elif 'reranked_documents' in state and state['reranked_documents']:
             # Fallback to reranked documents if debate summary is not useful
             context = "\n\n".join([doc.page_content for doc in state['reranked_documents']])
             print("Using reranked documents for synthesis (aggregated summary not available/useful).")
        else:
             print("No relevant context found in state for synthesis.")

    elif 'reranked_documents' in state and state['reranked_documents']:
        # Use reranked documents if no debate results
        context = "\n\n".join([doc.page_content for doc in state['reranked_documents']])
        print("Using reranked documents for synthesis.")

    elif 'retrieved_documents' in state and state['retrieved_documents']:
         # Fallback to initial retrieved documents if no reranked docs
         context = "\n\n".join([doc.page_content for doc in state['retrieved_documents']])
         print("Using initial retrieved documents for synthesis.")

    else:
        print("No documents or aggregated context found in state for synthesis.")
        # If no context, the LLM might still be able to answer based on its own knowledge
        # or indicate it cannot answer. Provide an empty context.
        context = ""


    synth_chain = SYNTHESIS_PROMPT | llm

    try:
        # If context is empty, the prompt will effectively ask the LLM to answer without context
        final_answer = synth_chain.invoke({"question": question, "context": context})
        print("Final answer generated.")
    except Exception as e:
        print(f"Error during synthesis LLM call: {e}")
        final_answer = f"Error generating answer: {e}"

    return {"final_answer": final_answer}

# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example synthesis...")
    # This component requires a loaded LLM and relevant context (documents or aggregated summary) in the state
    print("LLM and context not loaded for example. Skipping synthesis test.")
