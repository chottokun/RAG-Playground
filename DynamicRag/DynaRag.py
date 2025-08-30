import os
import streamlit as st
import configparser
import re
import sys

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 共通コンポーネントとLangChainの必要なモジュールをインポート
from shared_components.model_loader.load_llm import load_llm
from shared_components.pdf_processor import PDFProcessor
from langchain.prompts import PromptTemplate

# ---------- Configurations ----------
CONFIG_PATH = 'DynamicRag/config.ini'
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

TOP_N = config.getint('vectorstore', 'TOP_N', fallback=20)

# ---------- Prompts ----------
RERANKER_PROMPT = PromptTemplate(
    input_variables=["query", "docs_list"],
    template="""You are an expert at dynamically generating document identifiers to answer a given query.
I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.
Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.
Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.
If no documents are required to answer the query, output "None".
Output the identifiers as a comma-separated list, e.g., [1], [2] or "None" if no documents are needed.
Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text.

Query: {query}
Retrieved Content:
{docs_list}
"""
)

GENERATOR_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are an intelligent assistant that uses retrieved knowledge to answer user queries accurately and concisely.
Use the provided [Retrieved Content] to generate responses.
If the Retrieved Content is None, generate an answer based on your own knowledge.
If the information is insufficient or you don't know the answer, state, "I cannot fully answer based on the available information. Please provide more details."

Accuracy: Base your answers on the retrieved content.
Conciseness: Keep answers brief and relevant.
Context Awareness: Ensure your responses align with the user's query.

Query: {question}
Retrieved Content:
{context}

Answer:"""
)

# ---------- Streamlit UI ----------
def main():
    st.title("DynamicRAG Demo")

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
        base_url=config.get('ollama', 'BASE_URL', fallback=None),
        temperature=0.2
    )

    question = st.text_input("Enter your question:")
    if st.button("Ask") and question:
        with st.spinner("Retrieving top documents..."):
            candidates = vectorstore.similarity_search(question, k=TOP_N)

        docs_list = "\n".join(f"[{i+1}] Content: {c.page_content}" for i, c in enumerate(candidates))

        with st.spinner("Reranking documents..."):
            reranker_chain = RERANKER_PROMPT | llm
            rerank_output = reranker_chain.invoke({"query": question, "docs_list": docs_list})

        try:
            if rerank_output.strip().lower() == 'none':
                indices = []
            else:
                indices = [int(num) for num in re.findall(r'\[(\d+)\]', rerank_output)]
        except Exception:
            st.error(f"Failed to parse reranker output: {rerank_output}")
            st.stop()

        selected_docs = [candidates[i-1] for i in indices if 1 <= i <= len(candidates)]

        st.subheader("Selected & Ordered Documents")
        if not selected_docs:
            st.write("No retrieved content needed; using LLM knowledge.")
        else:
            for idx, doc in zip(indices, selected_docs):
                with st.expander(f"Document {idx}"):
                    st.write(doc.page_content)

        context = "\n\n".join([doc.page_content for doc in selected_docs]) if selected_docs else "None"

        with st.spinner("Generating answer..."):
            gen_chain = GENERATOR_PROMPT | llm
            answer = gen_chain.invoke({"question": question, "context": context})

        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
