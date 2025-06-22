import streamlit as st
import configparser
from typing import List, Dict, Any

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

class RAGOrchestrator:
    def __init__(self, config: configparser.ConfigParser, config_path: str):
        self.config = config
        self.config_path = config_path
        self.llm_model = config.get('LLM', 'MODEL', fallback='llama3')
        self.temperature = config.getfloat('LLM', 'TEMPERATURE', fallback=0.0)
        self.knowledge_directory = config.get('vectorstore', 'KNOWLEDGE_DIRECTORY', fallback='./vectorstore_rag_plus_knowledge')
        self.application_directory = config.get('vectorstore', 'APPLICATION_DIRECTORY', fallback='./vectorstore_rag_plus_application')

        # Initialize LLM
        ollama_base_url = self.config.get('ollama', 'BASE_URL', fallback='http://localhost:11434')
        self.llm = OllamaLLM(base_url=ollama_base_url, model=self.llm_model)

        # Load embedding function
        from RAG_plus.components.dual_corpus_builder import DualCorpusBuilder
        self.builder = DualCorpusBuilder(config)
        self.embedding_function = self.builder.embedding_function

        # Load vectorstores
        self.knowledge_vectorstore = self.builder.load_knowledge_vectorstore()
        self.application_vectorstore = self.builder.load_application_vectorstore()

        if self.knowledge_vectorstore is None or self.application_vectorstore is None:
            st.error("Failed to load vectorstores. Please check your configuration and data.")
            st.stop()

    def retrieve_applications(self, retrieved_knowledge_docs: List[Document]) -> str:
        """Retrieves application examples for the given knowledge documents."""
        applications = []
        for doc in retrieved_knowledge_docs:
            knowledge_id = doc.metadata.get("id")
            if knowledge_id:
                # Filter by both knowledge_id and type in the DB query for efficiency
                app_docs = self.application_vectorstore.similarity_search(
                    query=" ",  # Dummy query as filtering is the main goal
                    k=5,
                    filter={
                        "$and": [
                            {"knowledge_id": {"$eq": knowledge_id}},
                            {"type": {"$eq": "application"}},
                        ]
                    },
                )
                if app_docs:
                    # Original logic took only the first found application example.
                    applications.append(app_docs[0].page_content)
        return "\n---\n".join(applications)

    def run(self, question: str, history: List[Dict[str, Any]], st_callback=None) -> dict:
        """Runs the RAG+ pipeline and also returns the evidence."""
        # 1. Knowledge Retrieval
        knowledge_retriever = self.knowledge_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10, "filter": {"type": "knowledge"}}
        )

        # 2. RAG+ chain
        prompt_template = PromptTemplate.from_template(
            """
            Answer the question based on the following context.

            [Knowledge]
            {knowledge}

            [Application Examples]
            {application}

            [Question]
            {question}
            """
        )

        rag_plus_chain = (
            {
                "knowledge": (lambda x: x["question"]) | knowledge_retriever,
                "question": RunnablePassthrough()
            }
            | RunnablePassthrough.assign(
                application=lambda x: self.retrieve_applications(x["knowledge"])
            )
            | {
                "question": lambda x: x["question"],
                "knowledge": lambda x: "\n---\n".join([doc.page_content for doc in x["knowledge"]]),
                "knowledge_list": lambda x: [doc.page_content for doc in x["knowledge"]],
                "application": lambda x: x["application"]
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        # 3. Run the chain
        # knowledge_retrieverで取得したドキュメントも返す
        knowledge_docs = knowledge_retriever.invoke(question)
        application_text = self.retrieve_applications(knowledge_docs)
        response = rag_plus_chain.invoke({"question": question})
        return {
            "answer": response,
            "knowledge": [doc.page_content for doc in knowledge_docs],
            "application": application_text.split("\n---\n") if application_text else []
        }
