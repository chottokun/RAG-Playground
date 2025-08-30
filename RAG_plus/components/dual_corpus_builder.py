import configparser
from typing import List
from langchain_core.documents import Document

# 共通コンポーネントとLLMローダーをインポート
from shared_components.pdf_processor import PDFProcessor
from shared_components.model_loader.load_llm import load_llm

class DualCorpusBuilder:
    def __init__(self, config: configparser.ConfigParser, pdf_processor: PDFProcessor):
        """
        コンストラクタでConfigParserとPDFProcessorのインスタンスを受け取る
        """
        self.config = config
        self.pdf_processor = pdf_processor
        self.llm = self._load_llm()
        self.knowledge_directory = config.get('vectorstore', 'KNOWLEDGE_DIRECTORY')
        self.application_directory = config.get('vectorstore', 'APPLICATION_DIRECTORY')

    def _load_llm(self):
        """LLMをロードする"""
        return load_llm(
            provider=self.config.get('LLM', 'PROVIDER'),
            model=self.config.get('LLM', 'MODEL'),
            base_url=self.config.get('ollama', 'BASE_URL', fallback=None)
        )

    def _classify_chunk(self, chunk_text: str) -> str:
        """LLMを使ってチャンクを 'knowledge' または 'application' に分類する"""
        # langchain_coreに依存しないように、プロンプトは単純なf-stringで構築
        prompt = f"""Classify the following text chunk as 'knowledge' or 'application'.
'knowledge': Definitions, theorems, principles.
'application': Worked examples, problems, case studies.
Respond with only the category name.
Text: "{chunk_text}"
Category:"""
        category = self.llm.invoke(prompt).strip().lower()
        return category

    def build_and_save_corpora(self):
        """
        PDFからチャンクを取得し、分類し、2つのVectorstoreに保存するメインロジック
        """
        print("Starting dual corpus build process...")

        # 1. PDFProcessorを使ってチャンクを取得
        all_chunks = self.pdf_processor.get_chunks_from_pdfs()
        if not all_chunks:
            print("No chunks were retrieved from PDFs. Aborting.")
            return False

        knowledge_chunks = []
        application_chunks = []

        # 2. 各チャンクを分類
        print(f"Classifying {len(all_chunks)} chunks...")
        for chunk in all_chunks:
            category = self._classify_chunk(chunk.page_content)
            if category == "knowledge":
                knowledge_chunks.append(chunk)
            elif category == "application":
                # TODO: 元の実装にあったように、知識と応用を紐付けるIDを追加する
                # chunk.metadata['knowledge_id'] = ...
                application_chunks.append(chunk)
            else:
                print(f"Warning: Unknown category '{category}' for chunk.")

        print(f"Classification complete. Knowledge chunks: {len(knowledge_chunks)}, Application chunks: {len(application_chunks)}")

        # 3. 分類したチャンクをそれぞれのVectorstoreに保存
        if knowledge_chunks:
            self.pdf_processor.create_vectorstore_from_chunks(knowledge_chunks, self.knowledge_directory)

        if application_chunks:
            self.pdf_processor.create_vectorstore_from_chunks(application_chunks, self.application_directory)

        print("Dual corpus build process finished.")
        return True
