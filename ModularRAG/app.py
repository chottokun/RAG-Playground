import streamlit as st
import configparser
import os
import sys
from typing import List # Import List

# Add the project root to sys.path to allow importing from components and ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 
import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]



from components.pdf_processor import PDFProcessor
from ModularRAG.orchestrator import RAGOrchestrator # Import RAGOrchestrator
from ModularRAG.shared_types import HistoryItem # Import HistoryItem

# --- Configuration Loading ---
config = configparser.ConfigParser()
config_path = 'ModularRAG/config.ini'
if os.path.exists(config_path):
    config.read(config_path)
else:
    st.error(f"Configuration file not found at {config_path}")
    st.stop()

# --- Initialize Components (using Streamlit caching) ---
@st.cache_resource
def get_pdf_processor(config_path: str):
    """Caches the PDFProcessor instance."""
    try:
        processor = PDFProcessor(config_path=config_path)
        return processor
    except Exception as e:
        st.error(f"Error initializing PDFProcessor: {e}")
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
    st.title("Modular RAG System Demo")

    # Initialize PDF Processor and Orchestrator
    processor = get_pdf_processor(config_path)
    orchestrator = get_rag_orchestrator(config, config_path)

    if processor is None or orchestrator is None:
        st.warning("System components failed to initialize. Check configuration and dependencies.")
        return

    # PDF Indexing Section
    st.sidebar.header("PDF Indexing")
    st.sidebar.info(f"Vector store directory: {config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore_modular')}")
    st.sidebar.info(f"PDF path(s): {config.get('pdf', 'PATH', fallback='pdfs/')}")

    if st.sidebar.button("Index PDF"):
        with st.spinner("Indexing documents... This may take a minute."):
            try:
                vectorstore = processor.index_pdfs()
                if vectorstore:
                    st.success("Indexing completed!")
                    # キャッシュクリアの直接的な方法がないため、ユーザーにリロードを促す
                    st.info("Please manually reload the page to refresh the vectorstore cache.")
                else:
                    st.warning("No PDF documents found or processed for indexing.")
            except Exception as e:
                st.error(f"Error during indexing: {e}")
                import traceback
                st.error(traceback.format_exc())


    # Check if vectorstore exists before allowing RAG execution
    vectorstore_exists = processor.load_vectorstore() is not None

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
                final_answer = orchestrator.run(question_input, st.session_state.history, st_callback=st_progress)

                st.subheader("Final Answer")
                st.write(final_answer)

                # Update history (optional)
                # st.session_state.history.append({"query": question_input, "answer": final_answer})

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
