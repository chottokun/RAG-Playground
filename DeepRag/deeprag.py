import streamlit as st
import configparser
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Tuple, Any

# 
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# ---------- Configurations ----------
config = configparser.ConfigParser()
config.read('config.ini')
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
EMBEDDING_MODEL = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
LLM_MODEL = config.get('llm', 'MODEL', fallback='gemma3:4b-it-qat')
PERSIST_DIRECTORY = config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore')
PDF_PATH = config.get('pdf', 'PATH', fallback='DeepRag/2502.01142v1.pdf')

# ---------- DeepRAG Components ----------

class DeepRAG:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        # Prompt templates
        self.decomp_prompt = PromptTemplate(
            input_variables=["question", "history"],
            template=(
                "You are a RAG system that decomposes queries step-by-step.\n"
                "History: {history}\n"
                "Main Question: {question}\n"
                "Generate the next atomic subquery or 'TERMINATE' to finish."
            )
        )
        self.answer_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Use the following context to answer the query.\n"
                "Context: {context}\n"
                "Query: {query}\n"
                "Answer concisely."
            )
        )

    def binary_tree_search(self, question: str, max_depth: int = 5) -> Tuple[str, List[dict]]:
        def recurse(q: str, depth: int, path: List[dict]) -> Any:
            if depth >= max_depth:
                return path
            # Generate subquery or TERMINATE
            chain = LLMChain(llm=self.llm, prompt=self.decomp_prompt)
            sub = chain.run(question=q, history=" ; ".join(p['subquery'] for p in path)).strip()
            if sub.upper() == "TERMINATE":
                return path
            # Atomic decision: retrieve
            docs = self.vectorstore.similarity_search(sub, k=3)
            context = "\n\n".join(d.page_content for d in docs)
            ans_chain = LLMChain(llm=self.llm, prompt=self.answer_prompt)
            intermediate = ans_chain.run(query=sub, context=context)
            new_step = {"subquery": sub, "retrieved": True, "answer": intermediate}
            return recurse(sub, depth+1, path + [new_step])

        final_path = recurse(question, 0, [])
        final_context = "\n\n".join(step['answer'] for step in final_path)
        final_answer = self.llm(f"Based on above, answer: {question}\nContext: {final_context}")
        return final_answer, final_path

# ---------- Utility: Load and index PDF ----------

def load_vectorstore(pdf_path: str) -> Chroma:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIRECTORY)
    return store

# ---------- Streamlit UI ----------

def main():
    st.title("DeepRAG: Step-by-Step RAG Demo")
    st.info(f"Ollama BASE_URL: {OLLAMA_BASE_URL}")
    debug_mode = st.checkbox("DEBUGモード（検索コンテキスト表示）")
    if st.button("Index PDF"):
        with st.spinner("Indexing document... This may take a minute."):
            store = load_vectorstore(PDF_PATH)
            st.success("Indexing completed!")
    store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    dr = DeepRAG(llm, store)

    question = st.text_input("Enter your question:")
    if st.button("Run DeepRAG") and question:
        with st.spinner("Running retrieval-augmented reasoning..."):
            answer, trace = dr.binary_tree_search(question)
            st.subheader("Trace Steps")
            for idx, step in enumerate(trace, 1):
                st.markdown(f"**Step {idx}**: {step['subquery']}")
                st.markdown(f"- Retrieved: {step['retrieved']}")
                st.markdown(f"- Intermediate Answer: {step['answer']}")
                if debug_mode:
                    docs = store.similarity_search(step['subquery'], k=3)
                    st.markdown("<details><summary>🔍 検索コンテキスト</summary>", unsafe_allow_html=True)
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"<pre>【Doc {i}】\n{doc.page_content[:1000]}</pre>", unsafe_allow_html=True)
                    st.markdown("</details>", unsafe_allow_html=True)
            st.subheader("Final Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
