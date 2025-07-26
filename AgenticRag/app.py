import sys
import os
import configparser
import streamlit as st

# プロジェクトルートをsys.pathに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 共通モジュールと、リファクタリングしたグラフビルダーをインポート
from shared_components.pdf_processor import PDFProcessor
from shared_components.model_loader.load_llm import load_llm
from AgenticRag.graph_builder import build_agentic_graph, AgenticState

# --- 設定 ---
config = configparser.ConfigParser()
config_path = 'AgenticRag/config.ini'
config.read(config_path)

LLM_PROVIDER = config.get('LLM', 'PROVIDER', fallback='ollama')
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback=None)
LLM_MODEL = config.get('ollama', 'MODEL')

# --- Streamlit UI ---

@st.cache_resource
def get_components(config_path: str):
    """PDFProcessorとLLMをロードしてキャッシュする"""
    processor = PDFProcessor(config_path)
    vectordb = processor.load_vectorstore()
    if vectordb is None:
        st.info("Vectorstore not found, indexing PDF...")
        vectordb = processor.index_pdfs()
        if vectordb is None:
            st.error("Failed to create Vectorstore.")
            st.stop()

    llm = load_llm(LLM_PROVIDER, model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return vectordb, llm

def main():
    st.title("Agentic RAG")

    # コンポーネントを取得
    vectordb, llm = get_components(config_path)

    # グラフを構築
    # st.cache_resourceはコンパイル済みグラフ(Runnable)に対応していない場合があるので、毎回構築する
    graph = build_agentic_graph()

    question = st.text_input("質問を入力してください")
    if st.button("実行") and question:

        initial_state: AgenticState = {
            "question": question,
            "vectordb": vectordb,
            "llm": llm
        }

        with st.spinner("Agentic RAG is thinking..."):
            result = graph.invoke(initial_state)

        st.subheader("Retrieval Agent Output")
        st.write(result.get("retriever", []))

        st.subheader("Evaluator Output")
        st.write(result.get("evaluator", []))

        st.subheader("Refined Query")
        st.write(result.get("refiner", "N/A"))

        st.subheader("Secondary Retrieval")
        st.write(result.get("refined_retriever", []))

        st.subheader("Final Answer")
        st.write(result.get("final_answer", "No answer could be generated."))

if __name__ == "__main__":
    main()
