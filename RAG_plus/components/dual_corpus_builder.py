import streamlit as st
import configparser
import os
import sys
from typing import List # Import List

# Add the project root to sys.path to allow importing from components and ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Adjust path for RAG_plus
sys.path.insert(0, project_root)

# 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
from model_loader.load_embedding import load_embedding_model

class DualCorpusBuilder:
    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.knowledge_directory = config.get('vectorstore', 'KNOWLEDGE_DIRECTORY', fallback='./vectorstore_rag_plus_knowledge')
        self.application_directory = config.get('vectorstore', 'APPLICATION_DIRECTORY', fallback='./vectorstore_rag_plus_application')
        self.pdf_path = config.get('pdf', 'PATH', fallback='pdfs/')
        self.embedding_model_name = config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small')
        self.embedding_function = load_embedding_model(self.embedding_model_name)
        print(f"[DEBUG] embedding_function type: {type(self.embedding_function)}")

    def load_knowledge_vectorstore(self):
        """Loads the knowledge vectorstore from disk."""
        try:
            if not os.path.exists(self.knowledge_directory):
                st.warning(f"Knowledge vectorstore directory not found: {self.knowledge_directory}")
                return None
            knowledge_vectorstore = Chroma(persist_directory=self.knowledge_directory, embedding_function=self.embedding_function)
            return knowledge_vectorstore
        except Exception as e:
            st.error(f"Error loading knowledge vectorstore: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def load_application_vectorstore(self):
        """Loads the application vectorstore from disk."""
        try:
            if not os.path.exists(self.application_directory):
                st.warning(f"Application vectorstore directory not found: {self.application_directory}")
                return None
            application_vectorstore = Chroma(persist_directory=self.application_directory, embedding_function=self.embedding_function)
            return application_vectorstore
        except Exception as e:
            st.error(f"Error loading application vectorstore: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    def build_dual_corpus(self):
        """Builds the dual corpus from PDF documents."""
        pdf_files = [f for f in os.listdir(self.pdf_path) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("No PDF documents found in the specified path.")
            return False

        knowledge_corpus = []
        application_corpus = []
        last_knowledge_id = None

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_path, pdf_file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)

            # --- Step 3: LLMベースの分類器を定義 ---
            ollama_base_url = self.config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
            llm_model = self.config.get('LLM', 'MODEL', fallback='gemma3:4b-it-qat')
            llm = OllamaLLM(base_url=ollama_base_url, model=llm_model)
            classification_prompt = PromptTemplate.from_template(
                """Classify the following text chunk as 'knowledge' or 'application'.
                'knowledge': Definitions, theorems, principles.
                'application': Worked examples, problems, case studies.
                Respond with only the category name.
                Text: "{chunk_text}"
                Category:"""
            )
            classifier_chain = classification_prompt | llm | StrOutputParser()

            # --- Step 4: 分類とマッピング ---

            for chunk in chunks:
                chunk_text = chunk.page_content
                category = classifier_chain.invoke({"chunk_text": chunk_text}).strip().lower()

                if category == "knowledge":
                    # 新しい知識IDを生成
                    knowledge_id = f"k_{uuid.uuid4()}"
                    last_knowledge_id = knowledge_id  # last_knowledge_idを更新

                    knowledge_corpus.append({
                        "doc_id": knowledge_id,
                        "text": chunk_text,
                        "metadata": {"type": "knowledge", "id": knowledge_id}
                    })

                elif category == "application" and last_knowledge_id:
                    # 応用例として、直前の知識IDと紐付ける
                    application_id = f"a_{uuid.uuid4()}"

                    application_corpus.append({
                        "doc_id": application_id,
                        "text": chunk_text,
                        "metadata": {"type": "application", "knowledge_id": last_knowledge_id}
                    })

        # --- ChromaDBに格納 ---
        try:
            knowledge_vectorstore = Chroma.from_texts(
                texts=[item["text"] for item in knowledge_corpus],
                metadatas=[item["metadata"] for item in knowledge_corpus],
                ids=[item["doc_id"] for item in knowledge_corpus],
                embedding=self.embedding_function,
                persist_directory=self.knowledge_directory
            )

            application_vectorstore = Chroma.from_texts(
                texts=[item["text"] for item in application_corpus],
                metadatas=[item["metadata"] for item in application_corpus],
                ids=[item["doc_id"] for item in application_corpus],
                embedding=self.embedding_function,
                persist_directory=self.application_directory
            )

            return True

        except Exception as e:
            st.error(f"Error storing data in ChromaDB: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
