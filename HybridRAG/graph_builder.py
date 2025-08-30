import os
import networkx as nx
from typing import List, Tuple
from langchain_core.documents import Document

# LLMに知識トリプレットを抽出させるためのプロンプト
# このプロンプトは、テキストから(head, relation, tail)の形式で情報を引き出すように指示します。
TRIPLET_EXTRACTION_PROMPT = """
You are an expert in knowledge graph construction.
From the provided text, extract structured information as a list of triplets.
Each triplet should be in the format: (head_entity, relationship, tail_entity).
- "head_entity" and "tail_entity" should be specific entities, like names, organizations, or concepts.
- "relationship" should describe the connection between them.
- Focus on the most important and clearly stated relationships.

Example:
Text: "Apple, a tech giant, was co-founded by Steve Jobs. He also served as its CEO."
Triplets:
[("Apple", "is a", "tech giant"), ("Steve Jobs", "co-founded", "Apple"), ("Steve Jobs", "served as CEO of", "Apple")]

Now, extract triplets from the following text:

Text: "{text}"
Triplets:
"""

class KnowledgeGraphBuilder:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.graph_path = config.get('graph', 'PATH', fallback='knowledge_graph.gml')

    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """LLMを使ってテキストからトリプレットを抽出する"""
        prompt = TRIPLET_EXTRACTION_PROMPT.format(text=text)
        response = self.llm.invoke(prompt)

        # LLMの出力をパースして (h, r, t) のリストに変換する
        # ここではevalを使うが、本番環境ではより安全なパース方法が望ましい
        try:
            # LLMが `[("h", "r", "t")]` のような文字列を返すと仮定
            triplets = eval(response)
            if isinstance(triplets, list):
                return triplets
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing LLM response: {e}\nResponse: {response}")
            return []
        return []

    def build_from_documents(self, documents: List[Document]) -> nx.Graph:
        """ドキュメントリストからナレッジグラフを構築する"""
        graph = nx.Graph()
        print("Building knowledge graph from documents...")
        for doc in documents:
            triplets = self._extract_triplets(doc.page_content)
            for head, relation, tail in triplets:
                graph.add_node(head.strip(), type='entity')
                graph.add_node(tail.strip(), type='entity')
                graph.add_edge(head.strip(), tail.strip(), label=relation.strip())
        print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph

    def save_graph(self, graph: nx.Graph):
        """グラフをファイルに保存する"""
        print(f"Saving graph to {self.graph_path}...")
        nx.write_gml(graph, self.graph_path)
        print("Graph saved.")

    def load_graph(self) -> nx.Graph:
        """ファイルからグラフをロードする"""
        if os.path.exists(self.graph_path):
            print(f"Loading graph from {self.graph_path}...")
            graph = nx.read_gml(self.graph_path)
            print("Graph loaded.")
            return graph
        return None
