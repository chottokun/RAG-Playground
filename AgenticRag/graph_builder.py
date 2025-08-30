from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import configparser

# 新しい場所からノード関数をインポート
from .nodes.retrieval import retrieve_node, refined_retrieve_node
from .nodes.evaluation import evaluator_node
from .nodes.refinement import refiner_node
from .nodes.synthesis import synthesizer_node

class AgenticState(TypedDict, total=False):
    question: str
    vectordb: object
    llm: object
    config: configparser.ConfigParser
    retriever: List[str]
    evaluator: List[str]
    refiner: str
    refined_retriever: List[str]
    final_answer: str

def build_agentic_graph():
    """Agentic RAGのグラフを構築して返す"""
    workflow = StateGraph(state_schema=AgenticState)

    # ノードを追加
    workflow.add_node("retriever_node", retrieve_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("refiner_node", refiner_node)
    workflow.add_node("refined_retriever_node", refined_retrieve_node)
    workflow.add_node("synthesizer_node", synthesizer_node)

    # エッジを定義
    workflow.set_entry_point("retriever_node")
    workflow.add_edge("retriever_node", "evaluator_node")
    workflow.add_edge("evaluator_node", "refiner_node")
    workflow.add_edge("refiner_node", "refined_retriever_node")
    workflow.add_edge("refined_retriever_node", "synthesizer_node")
    workflow.add_edge("synthesizer_node", END)

    return workflow.compile()
