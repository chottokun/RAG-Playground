import streamlit as st
import configparser
import os
import sys

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 共通コンポーネントと、これから作成するオーケストレーターをインポート
from shared_components.pdf_processor import PDFProcessor
from shared_components.model_loader.load_llm import load_llm
from DeepRag.orchestrator import DeepRAG

# ---------- Streamlit UI ----------

def main():
    st.title("DeepRAG: Step-by-Step RAG Demo")

    # --- Config and Component Initialization ---
    config = configparser.ConfigParser()
    config.read('DeepRag/config.ini')

    # PDFProcessorを使用してVectorstoreを準備
    processor = PDFProcessor(config_path='DeepRag/config.ini')

    st.sidebar.header("Indexing")
    if st.sidebar.button("Index PDF"):
        with st.spinner("Indexing documents..."):
            store = processor.index_pdfs()
            if store:
                st.sidebar.success("Indexing completed!")
            else:
                st.sidebar.error("Indexing failed.")

    # Vectorstoreのロード
    vectorstore = processor.load_vectorstore()
    if vectorstore is None:
        st.warning("Vectorstore not found. Please index PDFs first.")
        st.stop()

    # LLMとDeepRAGオーケストレーターのロード
    llm = load_llm(
        provider=config.get('llm', 'PROVIDER', fallback='ollama'),
        model=config.get('llm', 'MODEL'),
        base_url=config.get('ollama', 'BASE_URL', fallback=None)
    )
    dr = DeepRAG(llm, vectorstore, config)

    # --- RAG Execution ---
    st.header("Ask a Question")
    question = st.text_input("Enter your question:")
    debug_mode = st.checkbox("DEBUGモード（検索コンテキスト表示）")

    if st.button("Run DeepRAG") and question:
        with st.spinner("Running retrieval-augmented reasoning..."):
            answer, trace = dr.binary_tree_search(question)

            st.subheader("Trace Steps")
            for idx, step in enumerate(trace, 1):
                st.markdown(f"**Step {idx}**: {step['subquery']}")
                st.markdown(f"- Retrieved: {step['retrieved']}")
                st.markdown(f"- Intermediate Answer: {step['answer']}")
                if debug_mode:
                    docs = vectorstore.similarity_search(step['subquery'], k=3)
                    st.markdown("<details><summary>🔍 検索コンテキスト</summary>", unsafe_allow_html=True)
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"<pre>【Doc {i}】\n{doc.page_content[:1000]}</pre>", unsafe_allow_html=True)
                    st.markdown("</details>", unsafe_allow_html=True)

            st.subheader("Final Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
