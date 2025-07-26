import networkx as nx
from typing import List

import re

def _find_entities_in_query(query: str, graph: nx.Graph) -> List[str]:
    """
    質問文に含まれる単語が、グラフのノード名と一致するかを探す。
    （簡易的なエンティティリンキング）
    """
    # 句読点を除去し、小文字に変換してトークン化
    clean_query = re.sub(r'[^\w\s]', '', query.lower())
    query_tokens = set(clean_query.split())

    graph_nodes = set([node.lower() for node in graph.nodes()])

    found_nodes = query_tokens.intersection(graph_nodes)

    # 元のグラフのノード名（大文字小文字を保持）を返す
    # 複数のトークンが同じノードを指す場合があるのでsetでユニークにする
    original_case_nodes = set()
    node_map = {node.lower(): node for node in graph.nodes()}
    for node in found_nodes:
        original_case_nodes.add(node_map[node])

    return list(original_case_nodes)


def graph_retrieval_component(query: str, graph: nx.Graph) -> str:
    """
    質問に基づいてナレッジグラフを検索し、関連情報をテキストで返す。

    Args:
        query: ユーザーからの質問文。
        graph: 探索対象のnetworkxグラフ。

    Returns:
        グラフから抽出されたコンテキスト情報（文字列）。
    """
    if graph is None or graph.number_of_nodes() == 0:
        return "No knowledge graph available."

    # 1. 質問からエンティティを特定
    entities = _find_entities_in_query(query, graph)

    if not entities:
        return "No relevant entities found in the knowledge graph for this query."

    # 2. グラフを探索し、コンテキストを生成
    context_parts = []
    for entity in entities:
        # エンティティとその隣接ノードの関係をテキスト化
        for neighbor in graph.neighbors(entity):
            edge_data = graph.get_edge_data(entity, neighbor)
            relation = edge_data.get('label', 'is related to')
            context_parts.append(f"{entity} {relation} {neighbor}.")

    if not context_parts:
        return f"Found entity '{', '.join(entities)}' but no specific relationships were retrieved."

    return " ".join(context_parts)
