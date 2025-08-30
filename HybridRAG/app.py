import streamlit as st
import configparser
import os
import sys

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 共通コンポーネントとHybridRAGモジュールをインポート
from shared_components.pdf_processor import PDFProcessor
from shared_components.model_loader.load_llm import load_llm
from HybridRAG.graph_builder import KnowledgeGraphBuilder
from HybridRAG.orchestrator import HybridRAGOrchestrator

# --- 設定とコンポーネントのロード ---

@st.cache_resource
def load_components(config_path):
    """すべての主要コンポーネントをロードしてキャッシュする"""
    config = configparser.ConfigParser()
    config.read(config_path)

    # LLMのロード
    llm = load_llm(
        provider=config.get('LLM', 'PROVIDER'),
        model=config.get('LLM', 'MODEL'),
        base_url=config.get('ollama', 'BASE_URL', fallback=None)
    )

    # PDFProcessorとVectorstoreの準備
    pdf_processor = PDFProcessor(config_path)
    vectorstore = pdf_processor.load_vectorstore()
    if vectorstore is None:
        st.warning("Vectorstore not found. Please build it first from the sidebar.")

    # KnowledgeGraphBuilderとKnowledge Graphの準備
    kg_builder = KnowledgeGraphBuilder(llm, config)
    knowledge_graph = kg_builder.load_graph()
    if knowledge_graph is None:
        st.warning("Knowledge Graph not found. Please build it first from the sidebar.")

    return config, llm, pdf_processor, vectorstore, kg_builder, knowledge_graph

def main():
    st.title("HybridRAG: Vector + Knowledge Graph Demo")

    config_path = 'HybridRAG/config.ini'
    config, llm, pdf_processor, vectorstore, kg_builder, knowledge_graph = load_components(config_path)

    # --- サイドバー ---
    st.sidebar.header("Index Construction")
    if st.sidebar.button("Build Vectorstore"):
        with st.spinner("Indexing PDFs for Vectorstore..."):
            pdf_docs = pdf_processor.index_pdfs()
            if pdf_docs:
                st.sidebar.success("Vectorstore built successfully!")
            else:
                st.sidebar.error("Failed to build Vectorstore.")

    if st.sidebar.button("Build Knowledge Graph"):
        if vectorstore is None:
            st.sidebar.error("Please build the Vectorstore first.")
        else:
            with st.spinner("Building Knowledge Graph from documents... This may take a while."):
                # PDFProcessorから直接チャンクを取得
                doc_objects = pdf_processor.get_chunks_from_pdfs()
                if not doc_objects:
                    st.sidebar.error("No documents found to build the knowledge graph.")
                    st.stop()

                new_kg = kg_builder.build_from_documents(doc_objects)
                kg_builder.save_graph(new_kg)
                st.sidebar.success("Knowledge Graph built successfully!")
                # キャッシュをクリアして再ロードを促す
                st.cache_resource.clear()
                st.experimental_rerun()

    # --- メイン画面 ---
    if vectorstore is None or knowledge_graph is None:
        st.info("Please build both the Vectorstore and the Knowledge Graph using the sidebar buttons.")
        st.stop()

    # Orchestratorの準備
    orchestrator = HybridRAGOrchestrator(llm, vectorstore, knowledge_graph, config)

    st.header("Ask a Question")
    query = st.text_area("Enter your question about the financial documents:")

    if st.button("Run HybridRAG") and query:
        with st.spinner("Running HybridRAG..."):
            result = orchestrator.run(query)

        st.subheader("Final Answer")
        st.markdown(result['final_answer'])

        with st.expander("Show Contexts"):
            st.subheader("Vector Context")
            st.text(result['vector_context'])
            st.subheader("Graph Context")
            st.text(result['graph_context'])

if __name__ == "__main__":
    main()
