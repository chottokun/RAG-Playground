import sys
import os
import json
from typing import Dict, Any, List, Optional # Import Optional

# Add the project root to sys.path to allow importing from ModularRAG and model_loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# Assuming LLM returns string for agent outputs, JSON for aggregator
from langchain_core.output_parsers import JsonOutputParser
from shared_components.model_loader.load_llm import load_llm # Need to load LLM with different temperatures

# Prompt template for individual debate agents
AGENT_PROMPT = PromptTemplate(
    input_variables=["query", "document", "agent_id", "role", "agg_summary", "agg_explanation"],
    template="""
You are Agent {agent_id}, serving as a {role} in a multi-agent RAG debate.
Use only the provided document to produce evidence or challenge existing summaries.
If agg_summary is "None", provide your initial stance.
Otherwise, defend, revise, or challenge based on:
Aggregate Summary: {agg_summary}
Explanation: {agg_explanation}
Be factual and concise, focusing on unique insights from the document.

Query: {query}
Document:
{document}

Your response:
"""
)

# Prompt template for the aggregator
AGGREGATOR_PROMPT = PromptTemplate(
    input_variables=["query", "agent_summaries"],
    template="""
You are the central Aggregator in a multi-agent debate.
Given agent summaries with diverse perspectives, produce:
1) A final concise answer(s) for ambiguous queries.
2) Explanation of choices, discarding misinformation or noise.

Agent Summaries:
{agent_summaries}

Query: {query}

Respond with JSON:
{{
  "summary": "<final answer>",
  "explanation": "<reasoning>"
}}
"""
)

# Default diversity settings (can be overridden by config)
DEFAULT_AGENT_TEMPS = [0.7, 0.5, 0.3, 0.1]
DEFAULT_AGENT_ROLES = [
    "skeptical critic",
    "detail-oriented summarizer",
    "concise responder",
    "contextual cross-checker"
]
DEFAULT_NUM_AGENTS = 4
DEFAULT_MAX_ROUNDS = 2 # Keep rounds low for initial testing

def debate_and_aggregation_component(state: RAGState, num_agents: int = DEFAULT_NUM_AGENTS, max_rounds: int = DEFAULT_MAX_ROUNDS) -> Dict[str, Any]:
    """
    Conducts a multi-agent debate and aggregates the results to generate a final answer.

    Args:
        state: The current RAGState, must contain 'question', 'refined_retrieved_documents', 'llm', and 'config'.
        num_agents: Number of agents to use in the debate.
        max_rounds: Maximum number of debate rounds.

    Returns:
        A dictionary to update the state, containing 'final_answer' and 'debate_results'.
    """
    question = state.get('question')
    documents = state.get('refined_retrieved_documents', []) # Use documents from refined retrieval
    base_llm = state.get('llm')
    config = state.get('config') # Access config for LLM loading

    if question is None or base_llm is None or config is None:
        print("Error: Question, LLM, or Config not available in state for debate.")
        return {"final_answer": "Error: Required state information missing for debate.", "debate_results": []}

    if not documents:
        print("No documents available for debate. Generating answer without debate.")
        # Fallback: Generate answer without debate if no documents
        synth_prompt = PromptTemplate(
            template="Given the question, provide a concise answer. No documents were found.\nQuestion: {question}\nAnswer:",
            input_variables=["question"],
        )
        synth_chain = synth_prompt | base_llm
        try:
            final_answer = synth_chain.invoke({"question": question})
            return {"final_answer": final_answer, "debate_results": [{"agent": "Synthesizer (Fallback)", "output": "No documents found for debate."}]}
        except Exception as e:
            print(f"Fallback synthesis failed: {e}")
            return {"final_answer": "Error generating answer.", "debate_results": []}

    print(f"Starting multi-agent debate with {len(documents)} documents, {num_agents} agents, {max_rounds} rounds...")

    # Prepare document context for agents
    doc_context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])

    # Get agent roles and temperatures (can be from config or use defaults)
    # For simplicity, let's use defaults for now, but config override is possible
    agent_roles = DEFAULT_AGENT_ROLES[:num_agents]
    agent_temps = DEFAULT_AGENT_TEMPS[:num_agents]

    if len(agent_roles) < num_agents or len(agent_temps) < num_agents:
         print(f"Warning: Not enough default roles/temps for {num_agents} agents. Using available defaults.")
         num_agents = min(len(agent_roles), len(agent_temps))
         agent_roles = agent_roles[:num_agents]
         agent_temps = agent_temps[:num_agents]


    debate_outputs = []
    # Initialize agent_summaries with empty strings instead of None
    agent_summaries: Dict[int, Optional[str]] = {i+1: "" for i in range(num_agents)}
    agg_summary = "None"
    agg_explanation = "None"
    round_traces = [] # To store trace of each round

    # Simulate debate rounds
    for round_num in range(1, max_rounds + 1):
        print(f"--- Debate Round {round_num} ---")
        round_outputs: Dict[int, str] = {} # Explicitly type round_outputs
        for i in range(num_agents):
            agent_id = i + 1
            role = agent_roles[i]
            temp = agent_temps[i]
            print(f"  Agent {agent_id} ({role}, Temp: {temp}) is thinking...")

            # Load LLM with specific temperature for this agent
            try:
                llm_agent = load_llm(
                    config['LLM']['PROVIDER'],
                    model=config['LLM']['MODEL'],
                    base_url=config['ollama'].get('BASE_URL', '').strip(),
                    temperature=temp # Set temperature here
                )
            except Exception as e:
                 print(f"Error loading LLM for Agent {agent_id}: {e}. Skipping agent.")
                 round_outputs[agent_id] = f"Error loading LLM: {e}"
                 continue


            # Convert agent_id and temp to string for the prompt template
            agent_prompt_instance = AGENT_PROMPT.partial(
                 agent_id=str(agent_id),
                 role=role,
                 temp=str(temp) # Include temp in prompt for agent awareness
            )

            agent_chain = agent_prompt_instance | llm_agent

            try:
                agent_output = agent_chain.invoke({
                    "query": question,
                    "document": doc_context, # Provide all documents to each agent for context
                    "agg_summary": agg_summary,
                    "agg_explanation": agg_explanation
                })
                agent_summaries[agent_id] = f"[Agent {agent_id} ({role})]: {agent_output.strip()}"
                round_outputs[agent_id] = agent_output.strip()
                print(f"    Agent {agent_id} Output: {agent_output[:100]}...") # Show snippet
            except Exception as e:
                print(f"  Agent {agent_id} failed: {e}")
                round_outputs[agent_id] = f"Error during debate: {e}"

        # Aggregate outputs from this round, filtering out None values
        summaries_text = "\n---\n".join([summary for summary in agent_summaries.values() if summary is not None])

        # Aggregator step (using base_llm with default temp)
        print("--- Aggregating Debate Results ---")
        # Aggregator expects JSON output, so use JsonOutputParser
        aggregator_chain = AGGREGATOR_PROMPT | base_llm | JsonOutputParser()

        try:
            agg_output = aggregator_chain.invoke({"query": question, "agent_summaries": summaries_text})
            # Basic validation of aggregator output
            if not isinstance(agg_output, dict) or 'summary' not in agg_output or 'explanation' not in agg_output:
                 print("Warning: Aggregator output did not match expected JSON structure.")
                 agg_summary = str(agg_output) # Store raw output if parsing fails
                 agg_explanation = "Aggregator output format incorrect."
            else:
                agg_summary = agg_output.get("summary", "N/A")
                agg_explanation = agg_output.get("explanation", "N/A")
            print("Aggregator Summary:", agg_summary[:100], "...")
            print("Aggregator Explanation:", agg_explanation[:100], "...")

        except Exception as e:
            print(f"Aggregator failed: {e}")
            agg_summary = f"Error during aggregation: {e}"
            agg_explanation = "Aggregation process failed."
            # If aggregation fails, the debate might not converge, but we still record the trace

        round_traces.append({
            "round": round_num,
            "agent_outputs": round_outputs,
            "agg_summary": agg_summary,
            "agg_explanation": agg_explanation
        })

        # Optional: Add convergence criteria here to break early

    # After all rounds, the final answer is the last aggregated summary
    final_answer = agg_summary

    return {"final_answer": final_answer, "debate_results": round_traces}


# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example debate and aggregation...")
    # This component requires a loaded LLM, documents, and config in the state
    print("LLM, documents, and config not loaded for example. Skipping debate and aggregation test.")
