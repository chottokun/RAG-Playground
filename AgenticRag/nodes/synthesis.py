def synthesizer_node(state):
    """統合ノード"""
    llm = state["llm"]
    refiner = state["refiner"]
    docs = state["refined_retriever"]

    synth_prompt = (
        "You are a Response Synthesis Agent. Using the refined query and retrieved documents,\n"
        "generate a concise and factual answer. Discard irrelevant or low-scored content.\n"
        "Refined Query: {refiner}\nDocuments: {docs}\nAnswer:"
    )

    prompt = synth_prompt.format(refiner=refiner, docs="\n".join(docs))
    # TODO: llm.invokeを使うように統一する
    answer = llm(prompt)

    return {"final_answer": answer.strip()}
