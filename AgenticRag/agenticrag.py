import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from components.pdf_processor import PDFProcessor
from model_loader.load_llm import load_llm
from langgraph.graph import StateGraph, END
import configparser

from typing import TypedDict, List

class AgenticState(TypedDict, total=False):
    question: str
    vectordb: object
    llm: object
    retriever: List[str]
    evaluator: List[str]
    refiner: str
    refined_retriever: List[str]
    final_answer: str

# 設定
config = configparser.ConfigParser()
config.read('AgenticRag/config.ini')
LLM_PROVIDER = config.get('LLM', 'PROVIDER', fallback='ollama')
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
LLM_MODEL = config.get('ollama', 'MODEL', fallback='gemma3:4b-it-qat')
EMBEDDING_MODEL = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
PERSIST_DIR = config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore')
PDF_PATH = config.get('pdf', 'PATH', fallback='2501.09136v3.pdf')
TOP_N = 5  # initial retrieval count (NUM_DOCSと同義)

# ベクトルDBの初期化
@st.cache_resource
def get_vectordb():
    processor = PDFProcessor('AgenticRag/config.ini')
    vectordb = processor.load_vectorstore()
    if vectordb is None:
        vectordb = processor.index_pdfs()
    return vectordb

# LangGraphノード定義（関数ベース）
def retrieve_node(state):
    vectordb = state["vectordb"]
    question = state["question"]
    docs = vectordb.similarity_search(question, k=TOP_N)
    return {"retriever": [d.page_content for d in docs]}

def evaluator_node(state):
    llm = state["llm"]
    question = state["question"]
    docs = state["retriever"]
    eval_prompt = (
        "You are a Relevance Evaluation Agent. Given a question and a document,\n"
        "score the document's relevance on a scale from 1 to 5, and provide a short rationale.\n"
        "Question: {question}\nDocument: {document}\nResponse format: [score] rationale"
    )
    results = []
    for doc in docs:
        prompt = eval_prompt.format(question=question, document=doc)
        results.append(llm(prompt))
    return {"evaluator": results}

def refiner_node(state):
    llm = state["llm"]
    question = state["question"]
    evaluator = state["evaluator"]
    refiner_prompt = (
        "You are a Query Refinement Agent. Based on the question and evaluator feedback,\n"
        "generate a refined query to improve retrieval.\n"
        "Question: {question}\nEvaluator Output: {evaluator}\nRefined Query:"
    )
    prompt = refiner_prompt.format(question=question, evaluator="\n".join(evaluator))
    refined_query = llm(prompt)
    return {"refiner": refined_query.strip()}

def refined_retrieve_node(state):
    vectordb = state["vectordb"]
    refined_q = state["refiner"]
    docs = vectordb.similarity_search(refined_q, k=TOP_N)
    return {"refined_retriever": [d.page_content for d in docs]}

def synthesizer_node(state):
    llm = state["llm"]
    refiner = state["refiner"]
    docs = state["refined_retriever"]
    synth_prompt = (
        "You are a Response Synthesis Agent. Using the refined query and retrieved documents,\n"
        "generate a concise and factual answer. Discard irrelevant or low-scored content.\n"
        "Refined Query: {refiner}\nDocuments: {docs}\nAnswer:"
    )
    prompt = synth_prompt.format(refiner=refiner, docs="\n".join(docs))
    answer = llm(prompt)
    return {"final_answer": answer.strip()}

# グラフ構築
def build_agentic_graph(vectordb, llm):
    workflow = StateGraph(state_schema=AgenticState)
    workflow.add_node("retriever_node", retrieve_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("refiner_node", refiner_node)
    workflow.add_node("refined_retriever_node", refined_retrieve_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    workflow.set_entry_point("retriever_node")
    workflow.add_edge("retriever_node", "evaluator_node")
    workflow.add_edge("evaluator_node", "refiner_node")
    workflow.add_edge("refiner_node", "refined_retriever_node")
    workflow.add_edge("refined_retriever_node", "synthesizer_node")
    workflow.add_edge("synthesizer_node", END)

    return workflow.compile()

# Streamlit UI
def main():
    st.title("Agentic RAG (共通モジュール)")
    vectordb = get_vectordb()
    llm = load_llm(LLM_PROVIDER, model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    question = st.text_input("質問を入力してください")
    if st.button("実行") and question:
        graph = build_agentic_graph(vectordb, llm)
        state = {
            "question": question,
            "vectordb": vectordb,
            "llm": llm
        }
        result = graph.invoke(state)
        st.subheader("Retrieval Agent Output")
        for idx, doc in enumerate(result["retriever"], 1):
            st.markdown(f"**Doc {idx}:** {doc[:200]}...")
        st.subheader("Evaluator Output")
        st.write(result["evaluator"])
        st.subheader("Refined Query")
        st.write(result["refiner"])
        st.subheader("Secondary Retrieval")
        for idx, doc in enumerate(result["refined_retriever"], 1):
            st.markdown(f"**Doc {idx}:** {doc[:200]}...")
        st.subheader("Final Answer")
        st.write(result["final_answer"])

if __name__ == "__main__":
    main()
