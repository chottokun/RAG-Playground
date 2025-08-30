def retrieve_node(state):
    """最初の検索ノード"""
    vectordb = state["vectordb"]
    question = state["question"]
    config = state["config"]
    top_k = config.getint('vectorstore', 'TOP_K', fallback=5)
    docs = vectordb.similarity_search(question, k=top_k)
    return {"retriever": [d.page_content for d in docs]}

def refined_retrieve_node(state):
    """再検索ノード"""
    vectordb = state["vectordb"]
    refined_q = state["refiner"]
    config = state["config"]
    top_k = config.getint('vectorstore', 'TOP_K', fallback=5)
    docs = vectordb.similarity_search(refined_q, k=top_k)
    return {"refined_retriever": [d.page_content for d in docs]}
