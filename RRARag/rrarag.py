import streamlit as st
import streamlit as st
import configparser
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from collections import Counter, defaultdict
from math import exp, log
import os

# 
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# ---------- Configurations ----------
config = configparser.ConfigParser()
config.read('DeepRag/config.ini') # config.iniのパスをDeepRag/config.iniに変更
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
EMBEDDING_MODEL = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
LLM_MODEL = config.get('llm', 'MODEL', fallback='gemma3:4b-it-qat')
PERSIST_DIRECTORY = config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore') # ベクトルストアのディレクトリをDeepRag/deeprag.pyに合わせる
PDF_PATH = config.get('pdf', 'PATH', fallback='2505.03676v1.pdf') # PDFパスは元のまま
ALPHA = 1.0  # RSA pragmatic parameter
TOP_K = 10   # initial candidate docs
FINAL_K = 5  # top docs after RRA re-ranking

# ---------- Utility functions for RRA ----------
def tokenize(text: str):
    # simple whitespace tokenizer; consider more advanced splitting
    return text.lower().split()


def build_term_doc_weights(docs):
    # docs: list of str
    term_doc = []  # list of Counter
    for doc in docs:
        ctr = Counter(tokenize(doc))
        term_doc.append(ctr)
    return term_doc


def compute_RRA_scores(query, docs_texts, term_doc_weights):
    # docs_texts: list of str, term_doc_weights aligned
    # compute sparse query weights
    w_q = Counter(tokenize(query))
    # vocabulary: union of query terms and doc terms
    vocab = set(w_q.keys())
    for ctr in term_doc_weights:
        vocab.update(ctr.keys())

    # pre-transform function f(x) = 1 + x
    def f(x): return 1 + x

    # compute L0(d|t) for each term and doc
    # L0[d][t]
    L0 = defaultdict(dict)
    # for each term, normalize over docs
    for t in vocab:
        # compute numerator for each doc
        nums = [f(term_doc_weights[i].get(t, 0)) for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L0[i][t] = num / denom

    # compute S1(t|d) for each doc
    S1 = defaultdict(dict)
    for i in range(len(docs_texts)):
        # all terms t in vocab
        scores = {t: exp(ALPHA * log(L0[i][t] + 1e-9)) for t in vocab}
        Z_d = sum(scores.values())
        for t, sc in scores.items():
            S1[i][t] = sc / (Z_d if Z_d > 0 else 1.0)

    # compute L1(d|t) for each term
    L1 = defaultdict(dict)
    num_docs = len(docs_texts)
    P_d = 1.0 / num_docs
    # for each term, normalize over docs
    for t in vocab:
        # numerator for each doc: P(d)*S1(t|d)
        nums = [P_d * S1[i][t] for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L1[i][t] = num / denom

    # compute final score(q,d)
    scores = []
    for i in range(len(docs_texts)):
        sc = 0.0
        for t, w in w_q.items():
            sc += w * L1[i].get(t, 0.0)
        scores.append((i, sc))
    # sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores]

# ---------- Data loading & indexing ----------
@st.cache_resource
def load_and_index():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)
    # vectordb.persist() # persist()は不要なので削除
    texts = [chunk.page_content for chunk in chunks]
    return vectordb, texts

# ---------- Streamlit App ----------
def main():
    st.title("Rational Retrieval Acts (RRA) RAG Demo")
    # ベクトルストアのロード方法をDeepRag/deeprag.pyに合わせる
    # load_and_index()は初回インデックス作成用として残し、実行時はロードのみ行う
    if st.button("Index PDF"):
        with st.spinner("Indexing document... This may take a minute."):
            store = load_and_index()[0] # load_and_indexを呼び出してベクトルストアを作成
            st.success("Indexing completed!")

    # 実行時は既存のベクトルストアをロード
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    docs_texts = vectordb.get()['documents']

    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0) # OllamaLLMを使用
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
Answer concisely.
"""
    )

    question = st.text_input("Enter your question:")
    if st.button("Ask") and question:
        with st.spinner("Retrieving candidates..."):
            # initial retrieval via vector store
            candidates = vectordb.similarity_search(question, k=TOP_K)
            cand_texts = [doc.page_content for doc in candidates]
            term_doc_weights = build_term_doc_weights(cand_texts)
            ranked_idx = compute_RRA_scores(question, cand_texts, term_doc_weights)
            top_docs = [cand_texts[i] for i in ranked_idx[:FINAL_K]]

        # show top docs
        st.subheader("Top Documents after RRA re-ranking")
        for i, txt in enumerate(top_docs, 1):
            st.markdown(f"**Document {i}:** {txt[:300]}... ")

        # generate answer
        context = "\n\n".join(top_docs)
        chain = llm
        input_prompt = prompt.format(context=context, question=question)
        with st.spinner("Generating answer..."):
            answer = chain.invoke(input_prompt)
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
