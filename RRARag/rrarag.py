import streamlit as st
import configparser
from collections import Counter, defaultdict
from math import exp, log
import os
import sys

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 共通コンポーネントとLangChainの必要なモジュールをインポート
from shared_components.model_loader.load_llm import load_llm
from shared_components.pdf_processor import PDFProcessor
from langchain.prompts import PromptTemplate

# ---------- Configurations ----------
CONFIG_PATH = 'RRARag/config.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

ALPHA = 1.0  # RSA pragmatic parameter
TOP_K = 10   # initial candidate docs
FINAL_K = 5  # top docs after RRA re-ranking

# ---------- Utility functions for RRA (変更なし) ----------
def tokenize(text: str):
    return text.lower().split()

def build_term_doc_weights(docs):
    term_doc = [Counter(tokenize(doc)) for doc in docs]
    return term_doc

def compute_RRA_scores(query, docs_texts, term_doc_weights):
    w_q = Counter(tokenize(query))
    vocab = set(w_q.keys())
    for ctr in term_doc_weights:
        vocab.update(ctr.keys())

    def f(x): return 1 + x

    L0 = defaultdict(lambda: defaultdict(float))
    for t in vocab:
        nums = [f(term_doc_weights[i].get(t, 0)) for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L0[i][t] = num / denom

    S1 = defaultdict(lambda: defaultdict(float))
    for i in range(len(docs_texts)):
        scores = {t: exp(ALPHA * log(L0[i][t] + 1e-9)) for t in vocab}
        Z_d = sum(scores.values())
        for t, sc in scores.items():
            S1[i][t] = sc / (Z_d if Z_d > 0 else 1.0)

    L1 = defaultdict(lambda: defaultdict(float))
    num_docs = len(docs_texts)
    P_d = 1.0 / num_docs
    for t in vocab:
        nums = [P_d * S1[i][t] for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L1[i][t] = num / denom

    scores = []
    for i in range(len(docs_texts)):
        sc = sum(w * L1[i].get(t, 0.0) for t, w in w_q.items())
        scores.append((i, sc))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores]

# ---------- Streamlit App ----------
def main():
    st.title("Rational Retrieval Acts (RRA) RAG Demo")

    # PDFProcessorを使用してVectorstoreを準備
    pdf_processor = PDFProcessor(config_path=CONFIG_PATH)

    st.sidebar.header("Indexing")
    if st.sidebar.button("Index PDF"):
        with st.spinner("Indexing documents..."):
            store = pdf_processor.index_pdfs()
            if store:
                st.sidebar.success("Indexing completed!")
            else:
                st.sidebar.error("Indexing failed. Check logs for details.")

    # Vectorstoreのロード
    vectorstore = pdf_processor.load_vectorstore()
    if vectorstore is None:
        st.warning("Vectorstore not found. Please index PDFs first.")
        st.stop()

    # LLMのロード
    llm = load_llm(
        provider=config.get('LLM', 'PROVIDER'),
        model=config.get('LLM', 'MODEL'),
        base_url=config.get('ollama', 'BASE_URL', fallback=None)
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question.
Context:
{context}
Question:
{question}
Answer concisely."""
    )

    question = st.text_input("Enter your question:")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and re-ranking candidates..."):
            candidates = vectorstore.similarity_search(question, k=TOP_K)
            cand_texts = [doc.page_content for doc in candidates]
            term_doc_weights = build_term_doc_weights(cand_texts)
            ranked_idx = compute_RRA_scores(question, cand_texts, term_doc_weights)
            top_docs = [cand_texts[i] for i in ranked_idx[:FINAL_K]]

        st.subheader(f"Top {FINAL_K} Documents after RRA re-ranking")
        for i, txt in enumerate(top_docs, 1):
            with st.expander(f"Document {i}"):
                st.write(txt)

        context = "\n\n".join(top_docs)
        input_prompt = prompt.format(context=context, question=question)
        with st.spinner("Generating answer..."):
            answer = llm.invoke(input_prompt)
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
