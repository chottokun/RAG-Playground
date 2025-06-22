import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import configparser
from typing import List, Dict, Any, Optional
import ast # Import ast for safer parsing

# import torch  # 削除



# Assume common components are available in the parent directories
# Adjust imports if the structure is different
from components.pdf_processor import PDFProcessor
from model_loader.load_llm import load_llm
from langchain_chroma import Chroma # Use langchain_chroma as per PDFProcessor
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END

# --- Configuration Loading ---
config = configparser.ConfigParser()
config_path = 'DeepStepRAG/config.ini'
if os.path.exists(config_path):
    config.read(config_path)
else:
    st.error(f"Configuration file not found at {config_path}")
    st.stop()

# LLM Loading
try:
    llm_config = config['LLM']
    ollama_config = config['ollama']
    # embedding_config = config['embedding'] # Embedding config is used by PDFProcessor
    # vectorstore_config = config['vectorstore'] # Vectorstore config is used by PDFProcessor
    # pdf_config = config['pdf'] # PDF config is used by PDFProcessor
    query_decomp_config = config['query_decomposition']
    debate_config = config['debate_node']

    base_url = ollama_config.get('BASE_URL')
    if base_url is not None:
        base_url = base_url.strip()
    llm = load_llm(
        llm_config['PROVIDER'],
        model=llm_config['MODEL'],
        base_url=base_url
    )

    # Embedding model loading is handled within PDFProcessor now
    # embeddings = None # No longer needed here

    # VECTORSTORE_DIR and PDF_PATH are accessed via config in the functions that need them
    # ENABLE_DYNAMIC_DECOMPOSITION, NUM_AGENTS, MAX_ROUNDS are accessed within the respective node functions

except KeyError as e:
    st.error(f"Missing section or key in config.ini: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading configuration or models: {e}")
    st.stop()

# --- Vector Store Initialization ---
# This part needs to be handled carefully in Streamlit to avoid re-indexing
@st.cache_resource
def get_vectorstore(config_path: str) -> Optional[Chroma]:
    """Loads the existing vectorstore using PDFProcessor."""
    try:
        processor = PDFProcessor(config_path=config_path)
        vectorstore = processor.load_vectorstore()
        if vectorstore:
            st.success(f"Vector store loaded from {processor.persist_directory}")
        else:
             st.info(f"Vector store directory {processor.persist_directory} not found or empty. Please index the PDF.")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

def index_pdf(config_path: str):
    """Indexes the PDF using PDFProcessor."""
    st.info("Indexing PDF...")
    try:
        processor = PDFProcessor(config_path=config_path)
        vectorstore = processor.index_pdfs()
        if vectorstore:
             st.success(f"PDF indexed and vector store created at {processor.persist_directory}")
        else:
             st.warning("No PDF documents found or processed for indexing.")
    except Exception as e:
        st.error(f"Error indexing PDF: {e}")


# --- LangGraph State Definition ---
class GraphState(BaseModel):
    """Represents the state of our graph."""
    question: str = Field(description="The user's question")
    sub_queries: List[str] = Field(default=[], description="List of decomposed sub-queries")
    retrieved_documents: List[Document] = Field(default=[], description="Documents retrieved from vector store")
    evaluation_results: List[Dict[str, Any]] = Field(default=[], description="Evaluation results for retrieved documents")
    refined_query: str = Field(default="", description="Refined query after evaluation") # Added default
    refined_retrieved_documents: List[Document] = Field(default=[], description="Documents retrieved after refinement")
    debate_results: List[Dict[str, Any]] = Field(default=[], description="Results from the multi-agent debate")
    final_answer: str = Field(default="", description="The final generated answer") # Added default
    # Add any other state variables needed

# --- LangGraph Nodes ---

# Node 1: Query Decomposition
def query_decomposition_node(state: GraphState):
    st.subheader("Step: Query Decomposition")
    question = state.question
    sub_queries = []

    # Access ENABLE_DYNAMIC_DECOMPOSITION from config within the node
    enable_dynamic_decomposition = config.getboolean('query_decomposition', 'ENABLE_DYNAMIC_DECOMPOSITION', fallback=True)

    if enable_dynamic_decomposition:
        st.info("Dynamically deciding whether to decompose the query...")
        # Implement logic to decide if decomposition is needed
        # This could involve checking query length, complexity keywords, etc.
        # For demonstration, let's decompose if the question is long or contains "and", "or"
        if len(question.split()) > 10 or any(word in question.lower() for word in [" and ", " or "]):
            st.info(f"Query '{question}' is complex. Attempting decomposition.")
            # Use LLM to decompose the query
            decomp_prompt = PromptTemplate(
                template="Given the following question, break it down into a list of simpler sub-queries. If the question is simple, just return the original question as a single item in the list.\nQuestion: {question}\nSub-queries:",
                input_variables=["question"],
            )
            try:
                # Assuming LLM returns a string like "['query1', 'query2']" or just "query"
                # Need a robust way to parse LLM output into a list of strings
                raw_output = (decomp_prompt | llm | RunnablePassthrough()).invoke({"question": question}) # Use RunnablePassthrough for string output
                # Simple parsing attempt - replace with a proper parser if needed
                if isinstance(raw_output, str):
                    # Attempt to parse string representation of list or single query
                    raw_output = raw_output.strip()
                    if raw_output.startswith('[') and raw_output.endswith(']'):
                         try:
                             # Use ast.literal_eval for safer parsing than eval
                             sub_queries = ast.literal_eval(raw_output)
                             if not isinstance(sub_queries, list):
                                 sub_queries = [question] # Fallback
                         except:
                             sub_queries = [question] # Fallback
                    else:
                         sub_queries = [raw_output] # Treat as single query
                elif isinstance(raw_output, list):
                     sub_queries = raw_output
                else:
                     sub_queries = [question] # Fallback

                st.write("Decomposed Sub-queries:", sub_queries)

            except Exception as e:
                st.warning(f"Query decomposition failed: {e}. Proceeding with original query.")
                sub_queries = [question]
        else:
            st.info("Query is simple. No decomposition needed.")
            sub_queries = [question]
    else:
        st.info("Dynamic decomposition is disabled. Proceeding with original query.")
        sub_queries = [question]

    return {"sub_queries": sub_queries}

# Node 2: Retriever
def retriever_node(state: GraphState):
    st.subheader("Step: Retrieval")
    sub_queries = state.sub_queries
    retrieved_documents = []
    # Pass config_path to get_vectorstore
    vectorstore = get_vectorstore(config_path)

    if vectorstore is None:
        st.error("Vector store not available.")
        return {"retrieved_documents": []}

    st.info(f"Retrieving documents for {len(sub_queries)} queries...")
    for query in sub_queries:
        st.write(f"Searching for: '{query}'")
        try:
            # Perform retrieval for each sub-query
            docs = vectorstore.similarity_search(query, k=5) # Adjust k as needed
            retrieved_documents.extend(docs)
            st.write(f"Found {len(docs)} documents for '{query}'.")
        except Exception as e:
            st.warning(f"Retrieval failed for query '{query}': {e}")

    # Optional: Deduplicate or re-rank retrieved_documents if needed
    # For simplicity, just returning the combined list for now
    st.write(f"Total retrieved documents: {len(retrieved_documents)}")
    return {"retrieved_documents": retrieved_documents}

# Node 3: Evaluator
def evaluator_node(state: GraphState):
    st.subheader("Step: Evaluation")
    question = state.question
    retrieved_documents = state.retrieved_documents
    evaluation_results = []

    if not retrieved_documents:
        st.warning("No documents to evaluate.")
        return {"evaluation_results": []}

    st.info(f"Evaluating {len(retrieved_documents)} documents...")

    # Use LLM to evaluate each document's relevance to the original question
    eval_prompt = PromptTemplate(
        template="Given the original question and a document, rate the document's relevance to the question on a scale of 1 to 5 (1 being not relevant, 5 being highly relevant). Also, provide a brief reason for your rating.\nOriginal Question: {question}\nDocument: {document_content}\nOutput JSON: {{'relevance_score': int, 'reason': str}}",
        input_variables=["question", "document_content"],
    )

    eval_chain = eval_prompt | llm | JsonOutputParser()

    for i, doc in enumerate(retrieved_documents):
        st.write(f"Evaluating document {i+1}/{len(retrieved_documents)}")
        try:
            result = eval_chain.invoke({"question": question, "document_content": doc.page_content})
            evaluation_results.append({"document": doc, "evaluation": result})
            st.write(f"  Score: {result.get('relevance_score', 'N/A')}, Reason: {result.get('reason', 'N/A')}")
        except Exception as e:
            st.warning(f"Evaluation failed for document {i+1}: {e}")
            evaluation_results.append({"document": doc, "evaluation": {"relevance_score": 1, "reason": f"Evaluation failed: {e}"}}) # Default to low score on failure

    return {"evaluation_results": evaluation_results}

# Node 4: Refiner
def refiner_node(state: GraphState):
    st.subheader("Step: Query Refinement")
    question = state.question
    evaluation_results = state.evaluation_results

    if not evaluation_results:
        st.warning("No evaluation results to refine the query.")
        return {"refined_query": question}

    st.info("Refining query based on evaluation results...")

    # Prepare context for refinement prompt
    eval_context = "\n".join([
        f"Document {i+1} (Score: {res['evaluation'].get('relevance_score', 'N/A')}): {res['evaluation'].get('reason', 'N/A')}"
        for i, res in enumerate(evaluation_results)
    ])

    refine_prompt = PromptTemplate(
        template="Given the original question and feedback on the relevance of retrieved documents, generate a refined query that would lead to better search results.\nOriginal Question: {question}\nDocument Evaluation Feedback:\n{eval_context}\nRefined Query:",
        input_variables=["question", "eval_context"],
    )

    refine_chain = refine_prompt | llm # Assuming LLM returns a string for the refined query

    try:
        refined_query = refine_chain.invoke({"question": question, "eval_context": eval_context})
        st.write("Refined Query:", refined_query)
    except Exception as e:
        st.warning(f"Query refinement failed: {e}. Using original question as refined query.")
        refined_query = question

    return {"refined_query": refined_query}

# Node 5: Refined Retriever
def refined_retriever_node(state: GraphState):
    st.subheader("Step: Refined Retrieval")
    refined_query = state.refined_query
    refined_retrieved_documents = []
    # Pass config_path to get_vectorstore
    vectorstore = get_vectorstore(config_path)

    if vectorstore is None:
        st.error("Vector store not available.")
        return {"refined_retrieved_documents": []}

    st.info(f"Retrieving documents using refined query: '{refined_query}'")
    try:
        # Perform retrieval with the refined query
        docs = vectorstore.similarity_search(refined_query, k=5) # Adjust k as needed
        refined_retrieved_documents.extend(docs)
        st.write(f"Found {len(docs)} documents for '{refined_query}'.")
    except Exception as e:
        st.warning(f"Refined retrieval failed for query '{refined_query}': {e}")

    st.write(f"Total refined retrieved documents: {len(refined_retrieved_documents)}")
    return {"refined_retrieved_documents": refined_retrieved_documents}

# Node 6: Debate Node (Multi-Agent Debate & Aggregation)
def debate_node(state: GraphState):
    st.subheader("Step: Multi-Agent Debate & Aggregation")
    question = state.question
    documents = state.refined_retrieved_documents # Use documents from refined retrieval

    if not documents:
        st.warning("No documents available for debate.")
        # Fallback: Generate answer without debate if no documents
        st.info("Generating answer without debate due to no documents.")
        synth_prompt = PromptTemplate(
            template="Given the question, provide a concise answer. No documents were found.\nQuestion: {question}\nAnswer:",
            input_variables=["question"],
        )
        synth_chain = synth_prompt | llm
        try:
            final_answer = synth_chain.invoke({"question": question})
            return {"final_answer": final_answer, "debate_results": [{"agent": "Synthesizer (Fallback)", "output": "No documents found for debate."}]}
        except Exception as e:
            st.error(f"Fallback synthesis failed: {e}")
            return {"final_answer": "Error generating answer.", "debate_results": []}


    st.info(f"Starting multi-agent debate with {len(documents)} documents...")

    # Prepare document context for agents
    doc_context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])

    # Define agent roles and temperatures (can be from config)
    # For simplicity, hardcoding example roles and temps
    agent_roles = ["Critically analyze the documents and question.", "Summarize the key points relevant to the question.", "Provide a concise answer based on the documents.", "Cross-check information across documents."]
    agent_temps = [0.7, 0.5, 0.3, 0.6]

    # Access NUM_AGENTS and MAX_ROUNDS from config within the node
    num_agents = config.getint('debate_node', 'NUM_AGENTS', fallback=4)
    max_rounds = config.getint('debate_node', 'MAX_ROUNDS', fallback=2)


    if len(agent_roles) != num_agents or len(agent_temps) != num_agents:
         st.warning(f"Number of agent roles ({len(agent_roles)}) or temperatures ({len(agent_temps)}) does not match NUM_AGENTS ({num_agents}) specified in config. Using default roles/temps and adjusting agent count.")
         # Adjust agent count based on available roles/temps or config, whichever is smaller
         num_agents = min(len(agent_roles), len(agent_temps), num_agents)
         agent_roles = agent_roles[:num_agents]
         agent_temps = agent_temps[:num_agents]


    debate_outputs = []

    # Simulate debate rounds (simplified)
    for round_num in range(max_rounds):
        st.write(f"--- Debate Round {round_num + 1} ---")
        round_outputs = []
        for i in range(num_agents):
            role = agent_roles[i]
            temp = agent_temps[i]
            st.write(f"  Agent {i+1} ({role}, Temp: {temp}) is thinking...")

            agent_prompt = PromptTemplate(
                template="You are an agent with the role: {role}. Your task is to analyze the following documents and the user's question. Provide your perspective or answer based *only* on the information in the documents. Be mindful of your assigned temperature ({temp}).\n\nOriginal Question: {question}\n\nDocuments:\n{doc_context}\n\nYour response:",
                input_variables=["role", "temp", "question", "doc_context"],
            )

            # Create a temporary LLM instance with specific temperature if supported
            # Note: Langchain LLM temperature setting might vary by provider
            # For Ollama, temperature is often set during model loading or invoke
            # Let's assume invoke supports temperature for simplicity or use a separate LLM instance
            # temp_llm = load_llm(llm_config['PROVIDER'], model_name=llm_config['MODEL'], base_url=ollama_config.get('BASE_URL'), temperature=temp) # Example temperature setting

            agent_chain = agent_prompt | llm # Using the main LLM instance for simplicity

            try:
                agent_output = agent_chain.invoke({
                    "role": role,
                    "temp": temp,
                    "question": question,
                    "doc_context": doc_context
                })
                round_outputs.append({"agent": f"Agent {i+1} ({role})", "output": agent_output})
                st.write(f"    Agent {i+1} Output: {agent_output[:100]}...") # Show snippet
            except Exception as e:
                st.warning(f"  Agent {i+1} failed: {e}")
                round_outputs.append({"agent": f"Agent {i+1} ({role})", "output": f"Error during debate: {e}"})

        debate_outputs.extend(round_outputs) # Collect outputs from all rounds

        # Optional: Incorporate previous round's outputs into the next round's prompt
        # This makes it a true debate. For simplicity, we'll just run rounds independently here.

    st.write("--- Aggregating Debate Results ---")

    # Aggregation step
    debate_summary = "\n\n".join([f"{res['agent']}:\n{res['output']}" for res in debate_outputs])

    aggregator_prompt = PromptTemplate(
        template="You are an aggregator. Given the original question and the outputs from multiple agents who analyzed relevant documents, synthesize their perspectives into a single, coherent, and comprehensive final answer. Explain the reasoning based on the documents.\n\nOriginal Question: {question}\n\nAgent Outputs:\n{debate_summary}\n\nFinal Answer (incorporating reasoning):",
        input_variables=["question", "debate_summary"],
    )

    aggregator_chain = aggregator_prompt | llm

    try:
        final_answer = aggregator_chain.invoke({"question": question, "debate_summary": debate_summary})
        st.write("Final Answer Generated.")
    except Exception as e:
        st.error(f"Aggregation failed: {e}")
        final_answer = "Error generating final answer after debate."

    return {"debate_results": debate_outputs, "final_answer": final_answer}


# --- LangGraph Workflow Definition ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("query_decomposition", query_decomposition_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("refiner", refiner_node)
workflow.add_node("refined_retriever", refined_retriever_node)
workflow.add_node("debate_node", debate_node)

# Set entry point
workflow.set_entry_point("query_decomposition")

# Add edges
workflow.add_edge("query_decomposition", "retriever")
workflow.add_edge("retriever", "evaluator")
workflow.add_edge("evaluator", "refiner")
workflow.add_edge("refiner", "refined_retriever")
workflow.add_edge("refined_retriever", "debate_node")
workflow.add_edge("debate_node", END) # End after debate and final answer generation

# Compile the graph
app = workflow.compile()

# --- Streamlit UI ---
st.title("DeepStepRAG: Combined RAG Demo")

# PDF Indexing Section
st.sidebar.header("PDF Indexing")
# PDFProcessor uses config.ini for paths, so no need for separate inputs here unless overriding config
# pdf_path_input = st.sidebar.text_input("PDF Path", value=PDF_PATH)
# vectorstore_dir_input = st.sidebar.text_input("Vector Store Directory", value=VECTORSTORE_DIR)

if st.sidebar.button("Index PDF"):
    index_pdf(config_path) # Pass config path

# RAG Execution Section
st.header("Ask a Question")
question_input = st.text_area("Enter your question here:")

if st.button("Run DeepStepRAG"):
    if question_input:
        vectorstore = get_vectorstore(config_path) # Pass config path
        if vectorstore:
            st.info("Running RAG process...")
            # Initial state
            initial_state = GraphState(question=question_input)

            # Run the graph
            try:
                # Pass vectorstore and llm to the graph execution if needed,
                # or ensure nodes can access them (e.g., via st.session_state or global scope if appropriate)
                # For simplicity, assuming global access for now, but passing is safer in complex apps.

                for s in app.stream(initial_state):
                    st.write(s) # Display state changes as they happen

                st.success("RAG process completed.")

            except Exception as e:
                st.error(f"An error occurred during RAG execution: {e}")
                import traceback
                st.error(traceback.format_exc())

        else:
            st.warning("Vector store not loaded. Please index the PDF first.")
    else:
        st.warning("Please enter a question.")

# Display final answer (if available in state after stream finishes)
# This might require capturing the final state from the stream
# For now, the stream output itself shows the final state
