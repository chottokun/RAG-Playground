# RAG Architecture & Developer's Guide

## 1. Philosophy and Goals

This repository is designed for the rapid experimentation and comparison of various Retrieval-Augmented Generation (RAG) techniques. Our architectural philosophy is centered around three key principles:

1.  **Modularity**: Each RAG implementation should be self-contained, with its core logic separated from the UI and other boilerplate code. This allows for clear separation of concerns.
2.  **Reusability**: Common functionalities, such as PDF processing and model loading, are centralized into `shared_components`. This avoids code duplication and ensures consistency across all experiments.
3.  **Testability**: Every new component and RAG implementation must be accompanied by tests. A robust test suite using `pytest` ensures reliability and allows for confident refactoring and expansion.

## 2. Standard Directory Structure

Every new RAG implementation **must** follow this standardized directory structure.

```
(YourRAGName)/
  ├── app.py              # Streamlit UI logic.
  ├── orchestrator.py     # Core RAG workflow logic (or graph_builder.py for LangGraph).
  ├── components/         # Your RAG-specific components (or nodes/).
  │   ├── __init__.py
  │   └── ...
  ├── config.ini          # Configuration specific to your RAG.
  └── README.md           # Detailed explanation of your RAG implementation.
```

-   **`app.py`**: Handles all user interface elements. Its primary role is to gather user input, call the orchestrator, and display the results.
-   **`orchestrator.py`**: Contains the main class or functions that define the step-by-step logic of your RAG pipeline. It initializes components and executes the flow.
-   **`components/`**: Holds the building blocks of your RAG logic (e.g., a query decomposition component, a synthesis component). Each file should have a single, clear responsibility.

## 3. How to Add a New RAG Implementation

Follow these steps to add a new RAG technique called `MyNewRAG`.

### Step 1: Create the Directory Structure

Create the standard directory structure inside the repository root.
```bash
mkdir -p MyNewRAG/components
touch MyNewRAG/app.py MyNewRAG/orchestrator.py MyNewRAG/config.ini MyNewRAG/README.md MyNewRAG/components/__init__.py
```

### Step 2: Configure `config.ini`

Set up the configuration for your RAG. At a minimum, you will need:
```ini
[LLM]
PROVIDER = ollama
MODEL = your-llm-model

[ollama]
BASE_URL = http://localhost:11434

[embedding]
MODEL = intfloat/multilingual-e5-small

[vectorstore]
DIRECTORY = ./vectorstore_mynewrag

[pdf]
DIRECTORY = ./pdfs
```

### Step 3: Implement the Core Logic in `orchestrator.py`

Define your main orchestrator class. This class will use the shared components.

```python
# MyNewRAG/orchestrator.py
# (Import necessary components)

class MyNewRAGOrchestrator:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        # Initialize your components here

    def run(self, query: str):
        # 1. Use self.vectorstore.similarity_search(...)
        # 2. Call your custom components
        # 3. Generate and return the final answer
        pass
```

### Step 4: Build the UI in `app.py`

Use this boilerplate to get started quickly. It handles loading shared components and calling your orchestrator.

```python
# MyNewRAG/app.py
import streamlit as st
import configparser
import os
import sys

# Project root setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import shared and your specific components
from shared_components.pdf_processor import PDFProcessor
from shared_components.model_loader.load_llm import load_llm
from MyNewRAG.orchestrator import MyNewRAGOrchestrator

@st.cache_resource
def load_components(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    llm = load_llm(...)
    pdf_processor = PDFProcessor(config_path)
    vectorstore = pdf_processor.load_vectorstore()
    return llm, vectorstore

def main():
    st.title("MyNewRAG")
    config_path = 'MyNewRAG/config.ini'
    llm, vectorstore = load_components(config_path)

    if st.sidebar.button("Build Vectorstore"):
        # ... (implementation) ...

    if vectorstore:
        orchestrator = MyNewRAGOrchestrator(llm, vectorstore)
        query = st.text_area("Ask a question:")
        if st.button("Run") and query:
            result = orchestrator.run(query)
            st.write(result)

if __name__ == "__main__":
    main()
```

### Step 5: Write Tests

**This is not optional.** Create a new test file `tests/test_mynewrag.py`.

-   Write unit tests for each new function in your `components/` directory.
-   Write an integration test for your `orchestrator.py`'s `run` method.
-   Use `unittest.mock.MagicMock` to mock external dependencies like LLM calls and vectorstore searches. Refer to `tests/test_deeprag.py` for a good example.

```python
# tests/test_mynewrag.py
import pytest
from unittest.mock import MagicMock

# Import your components
from MyNewRAG.orchestrator import MyNewRAGOrchestrator

def test_my_new_component():
    # ... test your component logic ...
    pass

def test_orchestrator_run():
    mock_llm = MagicMock()
    mock_vectorstore = MagicMock()
    # Configure mocks...

    orchestrator = MyNewRAGOrchestrator(mock_llm, mock_vectorstore)
    result = orchestrator.run("test query")

    # Assert that the result is as expected
    assert result is not None
    mock_llm.invoke.assert_called()
```

## 4. Using Shared Components

### `PDFProcessor`
Handles PDF loading, chunking, and vectorstore creation/loading.

-   **Initialization**: `PDFProcessor(config_path='path/to/your/config.ini')`
    - It reads `[vectorstore]` and `[pdf]` sections from your config.
    - **CRITICAL**: You **must** provide the correct path to your RAG-specific `config.ini`. If the path is incorrect, the processor will not raise an error but will use fallback default values, which can lead to hard-to-debug issues like using the wrong vectorstore directory.
-   **Core Methods**:
    -   `.get_chunks_from_pdfs() -> List[Document]`: Reads PDFs from the directory specified in `config.ini` and returns a list of document chunks.
    -   `.create_vectorstore_from_chunks(chunks, persist_directory)`: Creates a vectorstore from a list of chunks at the specified directory.
-   **Convenience Method**:
    -   `.index_pdfs()`: A handy method that combines the two steps above, using the default `persist_directory` from your `config.ini`.
    -   `.load_vectorstore()`: Loads an existing vectorstore from the `persist_directory`.

### `load_llm`
Loads an LLM instance from various providers.

-   **Usage**: `load_llm(provider, model, **kwargs)`
    - Reads configuration from your `config.ini` (`[LLM]`, `[ollama]`, etc.).
    - Example: `load_llm(provider='ollama', model='gemma3:4b-it-qat', base_url='...')`

## 5. Common Pitfalls and Best Practices

### Configuration Management
- **Always use a dedicated `config.ini`**: Each RAG implementation in its own directory must have its own `config.ini`. Do not reference configurations from other RAG directories.
- **Use a `CONFIG_PATH` constant**: In your `app.py` or main script, define the path to your config file as a constant to avoid magic strings.
  ```python
  # Good practice
  CONFIG_PATH = 'MyNewRAG/config.ini'
  config.read(CONFIG_PATH)
  processor = PDFProcessor(config_path=CONFIG_PATH)
  ```

### Refactoring Legacy Code
When updating older RAG implementations to use the shared components, follow these steps:
1.  **Replace Imports**: Find any imports from the old `components/` or `model_loader/` directories and replace them with `from shared_components...`.
2.  **Use `PDFProcessor`**: Remove any custom code for `PyPDFLoader`, `RecursiveCharacterTextSplitter`, and `Chroma.from_documents`. Replace it with a single instance of `PDFProcessor`.
3.  **Use `load_llm`**: Replace any direct instantiation of `OllamaLLM` or other LLM classes with a call to `load_llm`.
4.  **Verify `config.ini` paths**: Ensure the `PDFProcessor` is initialized with the correct `config_path` as described above.
