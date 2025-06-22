import sys
import os
from typing import List, Any, TypedDict

# Add the project root to sys.path to allow importing from model_loader and shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from model_loader.load_llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import shared types
from RAG_plus.shared_types import RAGState
from typing import TypedDict

# Define the expected output structure for the LLM reflection
class ReflectionResult(TypedDict):
    decision: str # "sufficient" or "insufficient"
    reasoning: str
    critique: str # Detailed critique if insufficient

# Prompt template for the LLM reflection
REFLECTION_PROMPT = PromptTemplate(
    template="""You are an expert answer evaluator. Given the original user query, the conversation history, and the current answer, determine if the current answer is sufficient to fully address the original query.

If the answer is sufficient, respond with:
```json
{{
  "decision": "sufficient",
  "reasoning": "The answer fully addresses the query.",
  "critique": ""
}}
```

If the answer is insufficient, provide a detailed critique explaining why it's insufficient and what information is missing or needs clarification. Then respond with:
```json
{{
  "decision": "insufficient",
  "reasoning": "The answer is incomplete or inaccurate.",
  "critique": "The answer does not provide details on X or Y."
}}
```

Original Query: {original_query}
Conversation History: {history}
Current Answer: {current_answer}

Output JSON:
Strictly follow the above format and output only JSON. Do not include any explanations or extra text.
""",
    input_variables=["original_query", "history", "current_answer"],
)

def reflection_component(state: RAGState) -> RAGState:
    """
    Evaluates the sufficiency of the current answer using an LLM.

    Args:
        state: The current RAGState, expected to contain 'question', 'llm', 'history', and 'final_answer'.

    Returns:
        Updated RAGState with 'reflection_result' (ReflectionResult TypedDict).
    """
    original_query = state.get('question')
    llm = state.get('llm')
    history = state.get('history', [])
    current_answer = state.get('final_answer', "No answer generated yet.")

    if not original_query or not llm:
        print("Error: 'question' or 'llm' not found in RAGState for reflection.")
        state['reflection_result'] = ReflectionResult(decision="error", reasoning="Missing query or LLM", critique="")
        return state

    print(f"Performing reflection on answer for query: '{original_query}'")

    # Format history for the prompt
    formatted_history = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history]) if history else "None"

    parser = JsonOutputParser()
    chain = REFLECTION_PROMPT | llm | parser

    try:
        reflection_output = chain.invoke({
            "original_query": original_query,
            "history": formatted_history,
            "current_answer": current_answer
        })
        
        # Basic validation and casting
        if isinstance(reflection_output, dict) and 'decision' in reflection_output and 'reasoning' in reflection_output:
            state['reflection_result'] = ReflectionResult(
                decision=str(reflection_output['decision']),
                reasoning=str(reflection_output['reasoning']),
                critique=str(reflection_output.get('critique', ''))
            )
            print(f"Reflection decision: {state['reflection_result']['decision']}")
        else:
            print("Warning: LLM reflection output did not match expected structure. Assuming insufficient.")
            state['reflection_result'] = ReflectionResult(decision="insufficient", reasoning="LLM output format incorrect.", critique="Could not parse LLM reflection output.")

    except Exception as e:
        print(f"Error during reflection: {e}. Assuming insufficient.")
        state['reflection_result'] = ReflectionResult(decision="insufficient", reasoning=f"LLM reflection failed: {e}", critique="An error occurred during reflection.")
    
    return state

if __name__ == "__main__":
    # Example Usage (for testing)
    print("Running reflection_component example...")
    
    # Dummy setup for testing
    import configparser
    
    test_config = configparser.ConfigParser()
    test_config.read_dict({
        'LLM': {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'},
        'ollama': {'BASE_URL': 'http://localhost:11434'}
    })

    llm = load_llm(test_config['LLM']['PROVIDER'], model=test_config['LLM']['MODEL'], base_url=test_config['ollama']['BASE_URL'])

    # Test 1: Sufficient answer
    initial_state_sufficient: RAGState = RAGState(
        question="What is the capital of France?",
        history=[],
        llm=llm,
        final_answer="The capital of France is Paris.",
        config=test_config
    )
    updated_state_sufficient = reflection_component(initial_state_sufficient)
    print(f"\nSufficient Answer Reflection Result: {updated_state_sufficient.get('reflection_result')}")

    # Test 2: Insufficient answer
    initial_state_insufficient: RAGState = RAGState(
        question="What is the capital of France and its population?",
        history=[],
        llm=llm,
        final_answer="The capital of France is Paris.", # Missing population
        config=test_config
    )
    updated_state_insufficient = reflection_component(initial_state_insufficient)
    print(f"\nInsufficient Answer Reflection Result: {updated_state_insufficient.get('reflection_result')}")

    # Test 3: With history
    initial_state_history: RAGState = RAGState(
        question="Tell me more about its history.",
        history=[{"query": "What is the capital of France?", "answer": "Paris."}],
        llm=llm,
        final_answer="Paris has a long history.",
        config=test_config
    )
    updated_state_history = reflection_component(initial_state_history)
    print(f"\nHistory Reflection Result: {updated_state_history.get('reflection_result')}")
