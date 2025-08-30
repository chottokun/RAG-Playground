def refiner_node(state):
    """リファインノード"""
    llm = state["llm"]
    question = state["question"]
    evaluator = state["evaluator"]

    refiner_prompt = (
        "You are a Query Refinement Agent. Based on the question and evaluator feedback,\n"
        "generate a refined query to improve retrieval.\n"
        "Question: {question}\nEvaluator Output: {evaluator}\nRefined Query:"
    )

    prompt = refiner_prompt.format(question=question, evaluator="\n".join(evaluator))
    # TODO: llm.invokeを使うように統一する
    refined_query = llm.invoke(prompt)

    return {"refiner": refined_query.strip()}
