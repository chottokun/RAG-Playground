import sys
import os

# Add the project root to sys.path to allow importing from shared_components.model_loader and shared_types
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Adjust path for RAG_plus
sys.path.append(project_root)

from shared_components.model_loader.load_llm import load_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any, List, TypedDict, cast

# Import shared types
from ModularRAG.shared_types import RAGState, HistoryItem

# Define the expected output structure for the LLM evaluation
class EvaluationResult(TypedDict):
    decision: str
    reasoning: str
    components: List[Dict[str, Any]]

# Prompt template for the LLM evaluator
EVALUATOR_PROMPT = PromptTemplate(
    template="""You are an expert RAG system designer. Given a user query and conversation history, analyze the complexity, intent, and context. Based on this analysis and the available RAG components, determine the optimal sequence of components and their parameters to answer the query effectively.

Available components and their general purpose:
- query_decomposition: Breaks down complex queries into simpler sub-queries. Useful for multi-faceted questions.
- knowledge_retrieval: Searches the knowledge corpus for relevant documents based on a query. Requires 'query' and 'k' (number of results).
- application_retrieval: Retrieves application examples linked to knowledge documents. Requires 'knowledge_documents'.
- evaluation: Assesses the relevance of retrieved documents to the original question.
- refinement: Generates a refined query based on initial retrieval and evaluation results. Sets 'refined_query' in state.
- reflection: Evaluates the sufficiency of the current answer. Sets 'reflection_result' in state as 'sufficient' or 'insufficient'.
- multi_agent_debate: Uses multiple agents to debate and synthesize information from documents. Good for complex or ambiguous topics. Requires 'documents', 'num_agents', 'max_rounds'.
- aggregation: Combines outputs from multiple sources (e.g., debate agents) into a final answer. Requires 'inputs'.
- reranking: Re-orders retrieved documents based on specific criteria (e.g., RRA, dynamic reranking). Requires 'query', 'documents'.
- rag_plus_synthesis: Generates a final answer based on retrieved knowledge and application examples. Requires 'knowledge_context', 'application_context'.

Consider the following:
- If the query is simple and direct, a simple knowledge_retrieval and rag_plus_synthesis might suffice.
- If the query is complex or asks for comparison/synthesis of multiple points, query decomposition and/or multi-agent debate might be beneficial.
- If initial retrieval seems insufficient, evaluation, reflection, and refinement steps can improve results.
- If documents might contain conflicting information, multi-agent debate can help synthesize a robust answer.
- If the answer is insufficient, use reflection and refinement to improve the query and repeat the process (up to a loop limit).
- Always consider using 'application_retrieval' and 'rag_plus_synthesis' for RAG+ capabilities, especially for questions requiring application-aware reasoning.

Provide your decision in JSON format, including the overall 'decision' (a brief description), 'reasoning' for your choice, and a list of 'components' to execute in order, with their required 'params'. If a component doesn't require specific parameters beyond its input from the previous step, an empty 'params' object is fine.

Example Output:
```json
{{
  "decision": "simple_rag_plus_retrieval_synthesis",
  "reasoning": "The query is a simple fact lookup, leveraging RAG+ for better context.",
  "components": [
    {{"name": "knowledge_retrieval", "params": {{"k": 5}}}},
    {{"name": "application_retrieval", "params": {{}}}},
    {{"name": "rag_plus_synthesis", "params": {{}}}}
  ]
}}
```
Example Output for a complex query with refinement:
```json
{{
  "decision": "retrieve_reflect_refine_rag_plus",
  "reasoning": "Initial answer may be insufficient, so use reflection and refinement, then apply RAG+ synthesis.",
  "components": [
    {{"name": "knowledge_retrieval", "params": {{"k": 10}}}},
    {{"name": "application_retrieval", "params": {{}}}},
    {{"name": "rag_plus_synthesis", "params": {{}}}},
    {{"name": "reflection", "params": {{}}}},
    {{"name": "refinement", "params": {{}}}}
  ]
}}
```

User Query: {query}
Conversation History: {history}

Output JSON:
Strictly follow the above format and output only JSON. Do not include any explanations or extra text.
Base your output strictly on the provided user query and conversation history.
""",
    input_variables=["query", "history"],
)

def evaluate_query_and_history(query: str, history: List[HistoryItem], llm: Any) -> EvaluationResult:
    """
    Evaluates the query and history using an LLM to determine the optimal RAG component sequence.

    Args:
        query: The user's current query.
        history: A list of previous query/answer pairs (as dictionaries).
        llm: The language model to use for evaluation.

    Returns:
        A dictionary representing the LLM's decision on the component sequence.
    """
    # JsonOutputParser with TypedDict doesn't automatically validate/cast,
    # so we'll parse to dict and rely on the return type hint and basic validation.
    parser = JsonOutputParser()

    chain = EVALUATOR_PROMPT | llm | parser

    # Format history for the prompt
    formatted_history = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history]) if history else "None"

    try:
        evaluation_result = chain.invoke({"query": query, "history": formatted_history})
        # Basic validation of the output structure
        if not isinstance(evaluation_result, dict) or 'components' not in evaluation_result or not isinstance(evaluation_result['components'], list):
             print("Warning: LLM output did not match expected structure. Returning empty components list.")
             # Ensure fallback matches the TypedDict structure
             return {"decision": "parsing_error", "reasoning": "LLM output format incorrect.", "components": []}

        # Although EvaluationResult is a TypedDict, returning a dict that matches its structure is acceptable.
        # Pylance might still warn, but runtime will be fine.
        return evaluation_result
    except Exception as e:
        print(f"Error during LLM evaluation: {e}")
        # Fallback to a simple sequence on error
        # Ensure fallback matches the TypedDict structure and cast for type checker
        return cast(EvaluationResult, {
            "decision": "error_fallback",
            "reasoning": f"LLM evaluation failed: {e}. Falling back to simple retrieval and synthesis.",
            "components": [
                {"name": "knowledge_retrieval", "params": {"k": 5}}, # Changed from retrieval
                {"name": "application_retrieval", "params": {}}, # Added
                {"name": "rag_plus_synthesis", "params": {}} # Changed from synthesis
            ]
        })

if __name__ == "__main__":
    # Example Usage (for testing)
    # This part would typically be run from the orchestrator
    print("Running example evaluation...")
    # Assuming you have a way to load your LLM here for testing
    # Example:
    # from shared_components.model_loader.load_llm import load_llm
    # llm_provider = "ollama" # Or read from config
    # llm_model = "gemma3:4b-it-qat" # Or read from config
    # llm_base_url = "http://localhost:11434" # Or read from config
    # test_llm = load_llm(llm_provider, model=llm_model, base_url=llm_base_url)

    # if 'test_llm' in locals():
    #     test_query = "What are the key differences between Agentic RAG and DeepStepRAG?"
    #     test_history = [] # Or some example history
    #     result = evaluate_query_and_history(test_query, test_history, test_llm)
    #     import json
    #     print(json.dumps(result, indent=2))
    # else:
    print("LLM not loaded for example. Skipping evaluation test.")
