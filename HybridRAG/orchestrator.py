import os
import sys

# プロジェクトルートをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# コンポーネントをインポート
from HybridRAG.components.graph_retrieval import graph_retrieval_component
# from HybridRAG.components.synthesis import synthesize_final_answer_with_hybrid_context

# synthesisコンポーネントはまだないので、一時的にここでロジックを実装
def synthesize_final_answer_with_hybrid_context(llm, question, vector_context, graph_context):
    prompt = f"""
You are an expert Q&A system. Use the following two sources of information to answer the question.
1. Vector-based Context: Provides broad, semantic information.
2. Graph-based Context: Provides structured, entity-relationship information.

Combine insights from both contexts to provide a comprehensive and accurate answer.

Vector Context:
{vector_context}

Graph Context:
{graph_context}

Question: {question}

Final Answer:
"""
    return llm.invoke(prompt)


class HybridRAGOrchestrator:
    def __init__(self, llm, vectorstore, knowledge_graph):
        self.llm = llm
        self.vectorstore = vectorstore
        self.knowledge_graph = knowledge_graph

    def run(self, query: str):
        """HybridRAGのメイン実行フロー"""

        # 1. VectorRAG検索
        print("Performing VectorRAG retrieval...")
        vector_docs = self.vectorstore.similarity_search(query, k=3)
        vector_context = "\n\n".join([doc.page_content for doc in vector_docs])
        print(f"Vector context retrieved:\n---\n{vector_context[:200]}...\n---")

        # 2. GraphRAG検索
        print("Performing GraphRAG retrieval...")
        graph_context = graph_retrieval_component(query, self.knowledge_graph)
        print(f"Graph context retrieved:\n---\n{graph_context}\n---")

        # 3. 回答生成
        print("Synthesizing final answer...")
        final_answer = synthesize_final_answer_with_hybrid_context(
            self.llm,
            query,
            vector_context,
            graph_context
        )
        print("Final answer generated.")

        return {
            "final_answer": final_answer,
            "vector_context": vector_context,
            "graph_context": graph_context
        }
