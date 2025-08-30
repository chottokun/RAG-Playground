def evaluator_node(state):
    """評価ノード"""
    llm = state["llm"]
    question = state["question"]
    docs = state["retriever"]

    eval_prompt = (
        "You are a Relevance Evaluation Agent. Given a question and a document,\n"
        "score the document's relevance on a scale from 1 to 5, and provide a short rationale.\n"
        "Question: {question}\nDocument: {document}\nResponse format: [score] rationale"
    )

    results = []
    for doc in docs:
        prompt = eval_prompt.format(question=question, document=doc)
        # TODO: llm.invokeを使うように統一する
        results.append(llm.invoke(prompt))

    return {"evaluator": results}
