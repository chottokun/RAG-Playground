import sys
import os
import re
from typing import Dict, Any, List
from collections import Counter, defaultdict # Import Counter and defaultdict
from math import exp, log # Import exp and log

# Add the project root to sys.path to allow importing from ModularRAG
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from ModularRAG.shared_types import RAGState # Import RAGState for type hinting
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Prompt template for the LLM-based reranker (from DynamicRag)
LLM_RERANKER_PROMPT = PromptTemplate(
    input_variables=["query", "docs_list"],
    template="""
You are an expert at dynamically generating document identifiers to answer a given query.
I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.
Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.
Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.
If no documents are required to answer the query, output "None".
Output the identifiers as a comma-separated list, e.g., [1], [2] or "None" if no documents are needed.
Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text.

Query: {query}
Retrieved Content:
{docs_list}
"""
)

# ---------- Utility functions for RRA (Integrated from RRARag) ----------
def tokenize(text: str):
    # simple whitespace tokenizer; consider more advanced splitting
    return text.lower().split()


def build_term_doc_weights(docs):
    # docs: list of str
    term_doc = []  # list of Counter
    for doc in docs:
        ctr = Counter(tokenize(doc))
        term_doc.append(ctr)
    return term_doc


def compute_RRA_scores(query, docs_texts, term_doc_weights, alpha: float = 1.0):
    # docs_texts: list of str, term_doc_weights aligned
    # compute sparse query weights
    w_q = Counter(tokenize(query))
    # vocabulary: union of query terms and doc terms
    vocab = set(w_q.keys())
    for ctr in term_doc_weights:
        vocab.update(ctr.keys())

    # pre-transform function f(x) = 1 + x
    def f(x): return 1 + x

    # compute L0(d|t) for each term and doc
    # L0[d][t]
    L0 = defaultdict(dict)
    # for each term, normalize over docs
    for t in vocab:
        # compute numerator for each doc
        nums = [f(term_doc_weights[i].get(t, 0)) for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L0[i][t] = num / denom

    # compute S1(t|d) for each doc
    S1 = defaultdict(dict)
    for i in range(len(docs_texts)):
        # all terms t in vocab
        scores = {t: exp(alpha * log(L0[i][t] + 1e-9)) for t in vocab} # Use alpha parameter
        Z_d = sum(scores.values())
        for t, sc in scores.items():
            S1[i][t] = sc / (Z_d if Z_d > 0 else 1.0)

    # compute L1(d|t) for each term
    L1 = defaultdict(dict)
    num_docs = len(docs_texts)
    P_d = 1.0 / num_docs
    # for each term, normalize over docs
    for t in vocab:
        # numerator for each doc: P(d)*S1(t|d)
        nums = [P_d * S1[i][t] for i in range(len(docs_texts))]
        denom = sum(nums) if sum(nums) > 0 else 1.0
        for i, num in enumerate(nums):
            L1[i][t] = num / denom

    # compute final score(q,d)
    scores = []
    for i in range(len(docs_texts)):
        sc = 0.0
        for t, w in w_q.items():
            sc += w * L1[i].get(t, 0.0)
        scores.append((i, sc))
    # sort descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scores]


def reranking_component(state: RAGState, method: str = "llm", top_k: int = 5, alpha: float = 1.0) -> Dict[str, Any]:
    """
    Re-ranks retrieved documents based on a specified method.

    Args:
        state: The current RAGState, must contain 'question', 'retrieved_documents', and 'llm' (for 'llm' method).
               May also need 'config' for RRA parameters (like alpha).
        method: The reranking method to use ('llm' or 'rra').
        top_k: The number of top documents to return after reranking.
        alpha: The alpha parameter for the RRA method.

    Returns:
        A dictionary to update the state, containing 'reranked_documents'.
    """
    question = state.get('question')
    retrieved_documents = state.get('retrieved_documents', [])
    llm = state.get('llm')
    config = state.get('config') # Might be needed for RRA parameters

    if question is None or not retrieved_documents:
        print("Error: Question or retrieved documents not available in state for reranking.")
        return {"reranked_documents": retrieved_documents[:top_k]} # Return initial top_k as fallback

    print(f"Performing reranking using method: {method}")

    reranked_docs: List[Document] = []

    if method == "llm":
        if llm is None:
            print("Error: LLM not available in state for LLM reranking.")
            return {"reranked_documents": retrieved_documents[:top_k]} # Fallback

        # Format docs for LLM reranker
        docs_list = "\n".join(f"[{i+1}] Content: {doc.page_content}" for i, doc in enumerate(retrieved_documents)) # Simplified format

        reranker_chain = LLM_RERANKER_PROMPT | llm
        try:
            rerank_output = reranker_chain.invoke({"query": question, "docs_list": docs_list})
            print("LLM Reranker Output:", rerank_output)

            # Parse identifiers
            if rerank_output.strip().lower() == 'none':
                indices = []
            else:
                indices = [int(num) for num in re.findall(r"\[(\d+)\]", rerank_output)]

            # Select documents based on parsed indices
            # Adjust indices to be 0-based and within bounds
            valid_indices = [i - 1 for i in indices if 1 <= i <= len(retrieved_documents)]
            # Ensure uniqueness and order based on LLM output
            unique_valid_indices = []
            for idx in valid_indices:
                if idx not in unique_valid_indices:
                    unique_valid_indices.append(idx)

            reranked_docs = [retrieved_documents[i] for i in unique_valid_indices[:top_k]]
            print(f"Selected {len(reranked_docs)} documents after LLM reranking.")

        except Exception as e:
            print(f"Error during LLM reranking: {e}. Using initial top_k documents.")
            reranked_docs = retrieved_documents[:top_k] # Fallback

    elif method == "rra":
        print("Performing RRA reranking.")
        docs_texts = [doc.page_content for doc in retrieved_documents]
        term_doc_weights = build_term_doc_weights(docs_texts)
        # Get alpha from config if available, otherwise use default parameter
        rra_alpha = alpha # Use the parameter passed to the component
        if config and 'reranking' in config and 'alpha' in config['reranking']:
             try:
                 rra_alpha = config.getfloat('reranking', 'alpha')
                 print(f"Using RRA alpha from config: {rra_alpha}")
             except ValueError:
                 print(f"Warning: Invalid 'alpha' value in config. Using default parameter: {rra_alpha}")


        ranked_indices = compute_RRA_scores(question, docs_texts, term_doc_weights, alpha=rra_alpha)
        reranked_docs = [retrieved_documents[i] for i in ranked_indices[:top_k]]
        print(f"Selected {len(reranked_docs)} documents after RRA re-ranking.")


    else:
        print(f"Unknown reranking method: {method}. Using initial top_k documents.")
        reranked_docs = retrieved_documents[:top_k] # Fallback

    # Ensure we return exactly top_k documents if possible, padding if necessary (though usually not needed after selection)
    # Or just return the selected ones up to top_k
    return {"reranked_documents": reranked_docs}

# Example Usage (for testing)
if __name__ == "__main__":
    print("Running example reranking...")
    # This component requires a loaded LLM (for 'llm' method) and retrieved documents in the state
    print("LLM and documents not loaded for example. Skipping reranking test.")
