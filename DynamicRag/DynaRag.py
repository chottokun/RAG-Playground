import os
import streamlit as st
import configparser
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------- Configurations ----------
config = configparser.ConfigParser()
config.read('config.ini')
OLLAMA_BASE_URL = config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
EMBEDDING_MODEL = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
LLM_MODEL = config.get('llm', 'MODEL', fallback='gemma3:4b-it-qat')
PERSIST_DIR = config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore')
PDF_PATH = config.get('pdf', 'PATH', fallback='2502.01142v1.pdf')
TOP_N = 20  # initial retrieval count

# ---------- Streamlit App: Load & Index ----------
@st.cache_resource
def load_vectorstore(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    return vectordb, chunks

# ---------- Prompt: Dynamic Reranker (Table 11) ----------
RERANKER_PROMPT = PromptTemplate(
    input_variables=["query", "docs_list"],
    template="""
You are an expert at dynamically generating document identifiers to answer a given query.
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

# ---------- Prompt: Retrieval-based Generator (Table 12) ----------
GENERATOR_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an intelligent assistant that uses retrieved knowledge to answer user queries accurately and concisely.

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
    st.title("DynamicRAG Demo with Streamlit")
    vectordb, chunks = load_vectorstore(PDF_PATH)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    question = st.text_input("Enter your question:")
    if st.button("Ask") and question:
        with st.spinner("Retrieving top documents..."):
            candidates = vectordb.similarity_search(question, k=TOP_N)
            docs_text = [doc.page_content for doc in candidates]

        # Format docs for reranker
        docs_list = "\n".join(f"[{i+1}] Title: {c.metadata.get('title','')} Content: {c.page_content}" for i, c in enumerate(candidates))
        reranker_chain = RERANKER_PROMPT | llm
        rerank_output = reranker_chain.invoke({"query": question, "docs_list": docs_list})

        # Parse identifiers
        try:
            if rerank_output.strip().lower() == 'none':
                indices = []
            else:
                import re
                indices = [int(num) for num in re.findall(r"\[(\d+)\]", rerank_output)]
        except Exception:
            st.error("Failed to parse reranker output: " + str(rerank_output))
            return

        # Select documents
        selected = [docs_text[i-1] for i in indices if 1 <= i <= len(docs_text)]
        st.subheader("Selected & Ordered Documents")
        if not selected:
            st.write("No retrieved content needed; using LLM knowledge.")
        else:
            for idx, doc in zip(indices, selected):
                st.markdown(f"**Doc {idx}:** {doc[:300]}...\n")

        # Generate final answer
        context = "\n\n".join(selected) if selected else None
        gen_chain = GENERATOR_PROMPT | llm
        with st.spinner("Generating answer..."):
            answer = gen_chain.invoke({"question": question, "context": context})
        st.subheader("Answer")
        st.write(answer)

if __name__ == "__main__":
    main()
