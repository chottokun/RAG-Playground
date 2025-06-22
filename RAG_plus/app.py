import streamlit as st
import configparser
import os
import sys
from typing import List # Import List
import asyncio

# Initialize asyncio event loop
asyncio.set_event_loop(asyncio.new_event_loop())

# Add the project root to sys.path to allow importing from components and ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust path for RAG_plus
sys.path.insert(0, project_root)

from RAG_plus.components.dual_corpus_builder import DualCorpusBuilder # Import DualCorpusBuilder
from RAG_plus.orchestrator import RAGOrchestrator # Import RAGOrchestrator
from RAG_plus.shared_types import HistoryItem # Import HistoryItem

# --- Configuration Loading ---
# RAG_plus/config.ini を明示的に利用
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config = configparser.ConfigParser()
if os.path.exists(config_path):
    config.read(config_path)
else:
    st.error(f"Configuration file not found at {config_path}")
    st.stop()

# --- Ensure vectorstore directories exist ---
knowledge_dir = config.get('vectorstore', 'KNOWLEDGE_DIRECTORY', fallback='./vectorstore_rag_plus_knowledge')
application_dir = config.get('vectorstore', 'APPLICATION_DIRECTORY', fallback='./vectorstore_rag_plus_application')
os.makedirs(knowledge_dir, exist_ok=True)
os.makedirs(application_dir, exist_ok=True)

# --- Initialize Components (using Streamlit caching) ---
@st.cache_resource
def get_dual_corpus_builder(_config: configparser.ConfigParser):
    """Caches the DualCorpusBuilder instance."""
    try:
        builder = DualCorpusBuilder(_config)
        return builder
    except Exception as e:
        st.error(f"Error initializing DualCorpusBuilder: {e}")
        return None

@st.cache_resource
def get_rag_orchestrator(_config: configparser.ConfigParser, config_path: str):
    """Caches the RAGOrchestrator instance."""
    try:
        orchestrator = RAGOrchestrator(_config, config_path)
        return orchestrator
    except Exception as e:
        st.error(f"Error initializing RAGOrchestrator: {e}")
        return None

# --- Streamlit UI ---
def main():
    st.title("RAG+ System Demo") # Changed title

    # Initialize Dual Corpus Builder and Orchestrator
    builder = get_dual_corpus_builder(config)
    orchestrator = get_rag_orchestrator(config, config_path)

    if builder is None or orchestrator is None:
        st.warning("System components failed to initialize. Check configuration and dependencies.")
        return

    # PDF Indexing Section
    st.sidebar.header("PDF Indexing for RAG+") # Changed header
    st.sidebar.info(f"Knowledge Vector store directory: {config.get('vectorstore', 'KNOWLEDGE_DIRECTORY', fallback='./vectorstore_rag_plus_knowledge')}")
    st.sidebar.info(f"Application Vector store directory: {config.get('vectorstore', 'APPLICATION_DIRECTORY', fallback='./vectorstore_rag_plus_application')}")
    st.sidebar.info(f"PDF path(s): {config.get('pdf', 'PATH', fallback='pdfs/')}")

    if st.sidebar.button("Index PDFs for RAG+"): # Changed button text
        with st.spinner("Building dual corpus... This may take a minute."):
            try:
                success = builder.build_dual_corpus()
                if success:
                    st.success("Dual corpus building completed!")
                    # キャッシュクリアの直接的な方法がないため、ユーザーにリロードを促す
                    st.info("Please manually reload the page to refresh the vectorstore cache.")
                else:
                    st.warning("No PDF documents found or processed for dual corpus building.")
            except Exception as e:
                st.error(f"Error during dual corpus building: {e}")
                import traceback
                st.error(traceback.format_exc())


    # Check if vectorstores exist before allowing RAG execution
    knowledge_store_exists = builder.load_knowledge_vectorstore() is not None
    application_store_exists = builder.load_application_vectorstore() is not None
    vectorstore_exists = knowledge_store_exists and application_store_exists

    # RAG Execution Section
    st.header("Ask a Question")
    question_input = st.text_area("Enter your question here:")

    # Simple history management (optional, for demonstration)
    if 'history' not in st.session_state:
        st.session_state.history = [] # Remove type hint here

    if st.button("Run RAG") and question_input:
        if not vectorstore_exists:
            st.warning("Vector store not found. Please index PDFs first.")
        else:
            st.info("Running RAG process...")
            progress_area = st.empty()  # 進捗表示用のエリア
            progress_msgs = []
            def st_progress(msg):
                print(msg)
                progress_msgs.append(msg)
                # 最新20件だけ表示（多すぎる場合の対策）
                progress_area.markdown('\n- '.join(progress_msgs[-20:]))
            try:
                # Run the orchestrator with the current question and history, passing st_progress
                result = orchestrator.run(question_input, st.session_state.history, st_callback=st_progress)

                st.subheader("Final Answer")
                st.write(result["answer"])

                with st.expander("根拠（Knowledge）を表示"):
                    for idx, k in enumerate(result["knowledge"], 1):
                        st.markdown(f"**Knowledge {idx}:**\n{k}\n")

                with st.expander("根拠（Application例）を表示"):
                    for idx, a in enumerate(result["application"], 1):
                        st.markdown(f"**Application {idx}:**\n{a}\n")

                # Update history (optional)
                # st.session_state.history.append({"query": question_input, "answer": result["answer"]})

            except Exception as e:
                st.error(f"An error occurred during RAG execution: {e}")
                import traceback
                st.error(traceback.format_exc())

    # Optional: Display history
    # if st.session_state.history:
    #     st.subheader("History")
    #     for item in reversed(st.session_state.history):
    #         st.markdown(f"**Q:** {item['query']}")
    #         st.markdown(f"**A:** {item['answer']}")
    #         st.markdown("---")


if __name__ == "__main__":
    main()
