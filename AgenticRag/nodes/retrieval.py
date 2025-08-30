def retrieve_node(state):
    """最初の検索ノード"""
    vectordb = state["vectordb"]
    question = state["question"]
    # TODO: TOP_Nをconfigから渡すようにする
    docs = vectordb.similarity_search(question, k=5)
    return {"retriever": [d.page_content for d in docs]}

def refined_retrieve_node(state):
    """再検索ノード"""
    vectordb = state["vectordb"]
    refined_q = state["refiner"]
    # TODO: TOP_Nをconfigから渡すようにする
    docs = vectordb.similarity_search(refined_q, k=5)
    return {"refined_retriever": [d.page_content for d in docs]}
