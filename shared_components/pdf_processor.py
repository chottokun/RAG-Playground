import os
import configparser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain_core.documents import Document

class PDFProcessor:
    def __init__(self, config_path='config.ini'):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.pdf_directory = self.config.get('pdf', 'DIRECTORY', fallback='./pdfs') # Assuming a new DIRECTORY option in config
        self.embedding_model_name = self.config.get('embedding', 'MODEL', fallback='intfloat/multilingual-e5-small').strip()
        self.persist_directory = self.config.get('vectorstore', 'DIRECTORY', fallback='./vectorstore')
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def get_chunks_from_pdfs(self) -> Optional[List[Document]]:
        from langchain_community.document_loaders import PyPDFLoader
        """Loads all PDFs and splits them into chunks, but does not save them."""
        all_chunks = []
        if not os.path.exists(self.pdf_directory):
            print(f"PDF directory not found: {self.pdf_directory}")
            # Raise a more specific error
            raise FileNotFoundError(f"PDF directory not found at: {self.pdf_directory}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        for filename in os.listdir(self.pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_directory, filename)
                print(f"Loading and processing {pdf_path}...")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    chunks = splitter.split_documents(docs)
                    all_chunks.extend(chunks)
                    print(f"Processed {len(chunks)} chunks from {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        if not all_chunks:
            print("No text could be extracted from the PDFs.")

        return all_chunks

    def create_vectorstore_from_chunks(self, chunks: List[Document], persist_directory: str) -> Optional[Chroma]:
        """Creates and saves a vectorstore from a list of chunks."""
        if not chunks:
            print("No chunks provided to create vectorstore.")
            return None

        print(f"Creating vectorstore with {len(chunks)} chunks at {persist_directory}...")
        store = Chroma.from_documents(chunks, self.embeddings, persist_directory=persist_directory)
        print("Vectorstore created.")
        return store

    def index_pdfs(self) -> Optional[Chroma]:
        """
        A convenience method that gets chunks from PDFs and creates a vectorstore
        at the default persist_directory.
        """
        try:
            chunks = self.get_chunks_from_pdfs()
            if not chunks:
                return None
            return self.create_vectorstore_from_chunks(chunks, self.persist_directory)
        except FileNotFoundError as e:
            print(e)
            return None

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Loads the existing vectorstore from the configured directory.
        """
        if not os.path.exists(self.persist_directory):
             print(f"Vectorstore directory not found: {self.persist_directory}")
             return None # Or raise an error

        print(f"Loading vectorstore from {self.persist_directory}...")
        store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        print("Vectorstore loaded.")
        return store

# Example usage (optional, for testing the component in isolation)
if __name__ == "__main__":
    # Ensure you have a config.ini and a pdfs/ directory with PDFs for testing
    processor = PDFProcessor()
    # To index PDFs:
    # vectorstore = processor.index_pdfs()
    # To load existing vectorstore:
    # vectorstore = processor.load_vectorstore()
    pass
