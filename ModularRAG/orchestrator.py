import sys
import os
import configparser
from typing import Dict, Any, List

# Add the project root to sys.path to allow importing from components and model_loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from model_loader.load_llm import load_llm
from components.pdf_processor import PDFProcessor
from ModularRAG.evaluator import evaluate_query_and_history, EvaluationResult # Import from our new evaluator module

# Import shared types
from ModularRAG.shared_types import RAGState, HistoryItem

# Import the modular components
from ModularRAG.components.query_decomposition import query_decomposition_component
from ModularRAG.components.retrieval import retrieval_component
from ModularRAG.components.evaluation import evaluation_component
from ModularRAG.components.refinement import refinement_component
from ModularRAG.components.reflection import reflection_component
from ModularRAG.components.debate_and_aggregation import debate_and_aggregation_component
from ModularRAG.components.reranking import reranking_component
from ModularRAG.components.synthesis import synthesis_component


# --- Configuration Loading ---
config = configparser.ConfigParser()
# Assuming a config file for the modular RAG, let's create one later or use a default
# For now, let's try to read a common config or define defaults
config_path = 'ModularRAG/config.ini' # Define a new config path for this modular RAG
if os.path.exists(config_path):
    config.read(config_path)
else:
    print(f"Warning: Configuration file not found at {config_path}. Using default settings.")
    # Define some default settings if config file is missing
    config['LLM'] = {'PROVIDER': 'ollama', 'MODEL': 'gemma3:4b-it-qat'}
    config['ollama'] = {'BASE_URL': 'http://localhost:11434'}
    config['embedding'] = {'MODEL': 'intfloat/multilingual-e5-small'}
    config['vectorstore'] = {'DIRECTORY': './vectorstore_modular'} # Use a new vectorstore directory
    config['pdf'] = {'PATH': 'pdfs/'} # Assuming a pdfs directory at the project root

# --- Component Mapping ---
# This dictionary will map component names (strings) to their corresponding functions or classes
# We will populate this as we modularize existing RAG components or create new ones.
COMPONENT_MAP: Dict[str, Any] = {
    "query_decomposition": query_decomposition_component,
    "retrieval": retrieval_component,
    "evaluation": evaluation_component,
    "refinement": refinement_component,
    "reflection": reflection_component,  # 追加
    "multi_agent_debate": debate_and_aggregation_component, # Using debate_and_aggregation for debate
    "aggregation": debate_and_aggregation_component, # Using debate_and_aggregation for aggregation (it handles both)
    "reranking": reranking_component,
    "synthesis": synthesis_component,
}

# --- State Management ---
# A simple dictionary to hold the state as components execute
# Moved to shared_types.py

# --- Orchestrator Logic ---
class RAGOrchestrator:
    def __init__(self, config: configparser.ConfigParser, config_path: str):
        self.config = config
        self.config_path = config_path # Store the config path
        self.llm = self._load_llm()
        self.vectorstore = self._load_vectorstore()
        self._populate_component_map() # Populate the map with actual component implementations
        self.component_defaults = self._load_component_defaults() # Load default parameters from config

    def _load_llm(self):
        llm_config = self.config['LLM']
        ollama_config = self.config['ollama']
        base_url = ollama_config.get('BASE_URL')
        if base_url is not None:
            base_url = base_url.strip()
        return load_llm(
            llm_config['PROVIDER'],
            model=llm_config['MODEL'],
            base_url=base_url
        )

    def _load_vectorstore(self):
        # Use PDFProcessor to load the vectorstore
        try:
            # Pass the config file path to PDFProcessor
            processor = PDFProcessor(config_path=self.config_path)
            vectorstore = processor.load_vectorstore()
            if vectorstore:
                print(f"Vector store loaded from {processor.persist_directory}")
            else:
                 print(f"Vector store directory {processor.persist_directory} not found or empty. Please index PDFs.")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None

    def _populate_component_map(self):
        # Map component names to their corresponding functions
        print("Populating component map...")
        global COMPONENT_MAP # Need to modify the global dictionary
        COMPONENT_MAP = {
            "query_decomposition": query_decomposition_component,
            "retrieval": retrieval_component,
            "evaluation": evaluation_component,
            "refinement": refinement_component,
            "reflection": reflection_component,  # 追加
            "multi_agent_debate": debate_and_aggregation_component, # Using debate_and_aggregation for debate
            "aggregation": debate_and_aggregation_component, # Using debate_and_aggregation for aggregation
            "reranking": reranking_component,
            "synthesis": synthesis_component,
        }
        print("Component map populated.")

    def run(self, question: str, history: List[HistoryItem] = [], max_loops: int = 3, st_callback=None) -> str:
        loop_count = 0
        current_question = question
        current_history = history.copy()
        final_answer = None

        while loop_count < max_loops:
            msg = f"Loop {loop_count+1}: Evaluating query: '{current_question}'"
            print(msg)
            if st_callback:
                st_callback(msg)
            evaluation_plan = evaluate_query_and_history(current_question, current_history, self.llm)
            msg = f"Execution plan: {evaluation_plan['decision']}"
            print(msg)
            if st_callback:
                st_callback(msg)
            msg = "Components to execute: " + ", ".join([comp['name'] for comp in evaluation_plan['components']])
            print(msg)
            if st_callback:
                st_callback(msg)

            # Initialize state
            state: RAGState = RAGState(question=current_question, history=current_history)
            state['llm'] = self.llm
            state['vectorstore'] = self.vectorstore
            state['config'] = self.config
            state['component_defaults'] = self.component_defaults

            for component_step in evaluation_plan['components']:
                component_name = component_step['name']
                plan_params = component_step.get('params', {})
                msg = f"Executing component: {component_name}"
                print(msg)
                if st_callback:
                    st_callback(msg)

                if component_name not in COMPONENT_MAP:
                    msg = f"Error: Component '{component_name}' not found in COMPONENT_MAP."
                    print(msg)
                    if st_callback:
                        st_callback(msg)
                    state['final_answer'] = f"Error: Component '{component_name}' not implemented."
                    break

                component_func = COMPONENT_MAP[component_name]
                default_params = self.component_defaults.get(component_name, {})
                final_params = {**default_params, **plan_params}

                if component_name == "synthesis":
                    final_params = {}

                try:
                    updated_state_dict = component_func(state, **final_params)
                    state.update(updated_state_dict)
                    msg = f"Component '{component_name}' executed successfully."
                    print(msg)
                    if st_callback:
                        st_callback(msg)
                except Exception as e:
                    msg = f"Error executing component '{component_name}': {e}"
                    print(msg)
                    if st_callback:
                        st_callback(msg)
                    import traceback
                    traceback.print_exc()
                    state['final_answer'] = f"Error executing component '{component_name}': {e}"
                    break

            final_answer = state.get('final_answer', None)

            # --- ループ継続判定 ---
            if state.get('reflection_result', None) == 'insufficient' and state.get('refined_query', None):
                msg = f"Reflection: Answer insufficient. Refining query: {state['refined_query']}"
                print(msg)
                if st_callback:
                    st_callback(msg)
                current_question = state['refined_query']
                current_history.append({"query": current_question, "answer": final_answer or ""})
                loop_count += 1
                continue
            else:
                break  # 十分な答えが得られた、または再質問不要

        return final_answer or "No final answer generated."

    def _load_component_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Loads default component parameters from the config file."""
        defaults: Dict[str, Dict[str, Any]] = {}
        print("--- Loading Component Defaults ---")
        print(f"Config sections found: {self.config.sections()}")
        for section in self.config.sections():
            print(f"Processing section: {section}")
            # Assuming section names match component names (case-insensitive check might be needed)
            # Exclude standard sections like LLM, ollama, embedding, vectorstore, pdf
            if section.lower() not in ['llm', 'ollama', 'embedding', 'vectorstore', 'pdf']:
                component_name = section.lower() # Use lowercase name for mapping
                print(f"Section '{section}' is a component section. Mapping to '{component_name}'.")
                defaults[component_name] = {}
                for key, value in self.config.items(section):
                    print(f"  Processing key '{key}' with value '{value}' in section '{section}'")
                    # Attempt to convert value to appropriate type (int, float, bool)
                    try:
                        if '.' in value:
                            converted_value = self.config.getfloat(section, key)
                            print(f"    Converted to float: {converted_value}")
                            defaults[component_name][key.lower()] = converted_value
                        elif value.lower() in ['true', 'false']:
                            converted_value = self.config.getboolean(section, key)
                            print(f"    Converted to boolean: {converted_value}")
                            defaults[component_name][key.lower()] = converted_value
                        else:
                            converted_value = self.config.getint(section, key)
                            print(f"    Converted to int: {converted_value}")
                            defaults[component_name][key.lower()] = converted_value
                    except ValueError:
                        print(f"    Could not convert value '{value}', keeping as string.")
                        defaults[component_name][key.lower()] = value # Keep as string if conversion fails
            else:
                print(f"Section '{section}' is a standard section, skipping.")
        print("--- Finished Loading Component Defaults ---")
        print("Loaded component defaults:", defaults)
        return defaults


# --- Main Execution ---
if __name__ == "__main__":
    # Example Usage
    # Need to pass config_path when creating the orchestrator instance
    orchestrator = RAGOrchestrator(config, config_path)

    # Example: Indexing PDF (should be done once)
    # You might want a separate script or UI for indexing
    # print("Checking/Indexing PDF...")
    # processor = PDFProcessor(config=config)
    # if processor.load_vectorstore() is None:
    #     processor.index_pdfs()

    # Example Query
    query = "What is the main idea of the paper?"
    # Example History (list of HistoryItem)
    history: List[HistoryItem] = [] # Example: [{"query": "Previous Q", "answer": "Previous A"}]

    print(f"\nRunning RAG for query: '{query}'")
    final_answer = orchestrator.run(query, history)

    print("\n--- Final Answer ---")
    print(final_answer)

    # Example with a different query
    # query_complex = "Compare the multi-agent debate approach with the RRA re-ranking method."
    # print(f"\nRunning RAG for query: '{query_complex}'")
    # final_answer_complex = orchestrator.run(query_complex, history)
    # print("\n--- Final Answer (Complex Query) ---")
    # print(final_answer_complex)
