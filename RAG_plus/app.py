import streamlit as st
import configparser
import os
import sys

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from shared_components.pdf_processor import PDFProcessor
from RAG_plus.components.dual_corpus_builder import DualCorpusBuilder
# Orchestratorも後で使う
# from RAG_plus.orchestrator import RAGPlusOrchestrator

@st.cache_resource
def load_pdf_processor(config_path):
    """PDFProcessorをロード"""
    try:
        return PDFProcessor(config_path)
    except FileNotFoundError as e:
        st.error(e)
        st.stop()

def main():
    st.title("RAG-plus Demo")
    config_path = 'RAG_plus/config.ini'

    config = configparser.ConfigParser()
    config.read(config_path)

    pdf_processor = load_pdf_processor(config_path)

    st.sidebar.header("Corpus Management")
    if st.sidebar.button("Build Dual Corpus"):
        with st.spinner("Building dual corpus... This may take a while as it classifies each chunk."):
            # DualCorpusBuilderにPDFProcessorのインスタンスを渡す
            builder = DualCorpusBuilder(config, pdf_processor)
            success = builder.build_and_save_corpora()
            if success:
                st.sidebar.success("Dual corpus built successfully!")
            else:
                st.sidebar.error("Failed to build dual corpus.")

    st.header("Ask a Question")

    # Vectorstoreのロードを試みる
    knowledge_vs_dir = config.get('vectorstore', 'KNOWLEDGE_DIRECTORY')
    application_vs_dir = config.get('vectorstore', 'APPLICATION_DIRECTORY')

    if not os.path.exists(knowledge_vs_dir) or not os.path.exists(application_vs_dir):
        st.info("Vectorstores not found. Please build the dual corpus first.")
        st.stop()

    # TODO: Orchestratorを実装し、質問応答フローをここに記述する
    st.markdown("Corpora are ready. The Q&A functionality is not yet implemented in this refactored version.")


if __name__ == "__main__":
    main()
