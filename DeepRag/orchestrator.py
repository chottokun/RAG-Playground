import configparser
from typing import List, Tuple, Any
from langchain.prompts import PromptTemplate # Orchestratorはlangchainに依存

import os
import sys
# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# コンポーネントをインポート
from DeepRag.components.decomposition import decomposition_component
from DeepRag.components.synthesis import synthesize_intermediate_answer, synthesize_final_answer

class DeepRAG:
    """リファクタリング後のOrchestrator"""
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        # プロンプトテンプレートをここで定義
        self.decomp_prompt = PromptTemplate(
            input_variables=["question", "history"],
            template=(
                "You are a RAG system that decomposes queries step-by-step.\n"
                "History: {history}\n"
                "Main Question: {question}\n"
                "Generate the next atomic subquery or 'TERMINATE' to finish."
            )
        )
        self.synthesis_prompt = PromptTemplate(
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

            # 1. Decompositionコンポーネントを呼び出す
            history_str = " ; ".join(p['subquery'] for p in path)
            sub_query = decomposition_component(self.llm, self.decomp_prompt, q, history_str)

            if sub_query.upper() == "TERMINATE":
                return path

            # 2. Retrieval
            docs = self.vectorstore.similarity_search(sub_query, k=3)
            context = "\n\n".join(d.page_content for d in docs)

            # 3. Synthesisコンポーネント（中間）を呼び出す
            intermediate_answer = synthesize_intermediate_answer(self.llm, self.synthesis_prompt, sub_query, context)

            new_step = {"subquery": sub_query, "retrieved": True, "answer": intermediate_answer}
            return recurse(sub_query, depth + 1, path + [new_step])

        # 再帰処理の開始
        final_path = recurse(question, 0, [])

        # 4. Synthesisコンポーネント（最終）を呼び出す
        intermediate_answers = [step['answer'] for step in final_path]
        final_answer = synthesize_final_answer(self.llm, question, intermediate_answers)

        return final_answer, final_path
