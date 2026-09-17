from typing import Optional, Any
from src.custom_reranker import final_score as heuristic_final_score


def final_score(row, llm=None) -> float:
    """Compute ranking score for a retrieved candidate document.

    Combines temporal and entity heuristic score (from E2 custom reranker).
    If an LLM wrapper is provided and configured for scoring, it can integrate semantic weighting.
    """
    # Base heuristic score from temporal proximity + NER match
    base_score = heuristic_final_score(row)

    # Optional LLM-assisted scoring if requested
    if llm is not None and hasattr(llm, "score_relevance"):
        try:
            llm_boost = llm.score_relevance(row.get("query", ""), row.get("text", ""))
            return base_score * (1.0 + float(llm_boost))
        except Exception:
            pass

    return base_score


def build_rag_prompt(query: str, retrieved_docs: Any, max_context_docs: int = 3) -> str:
    """Construct a grounded Question-Answering prompt from retrieved historical documents."""
    context_chunks = []
    # Handle DataFrame or list of dicts/records
    if hasattr(retrieved_docs, "iterrows"):
        doc_rows = [r for _, r in retrieved_docs.head(max_context_docs).iterrows()]
    elif isinstance(retrieved_docs, list):
        doc_rows = retrieved_docs[:max_context_docs]
    else:
        doc_rows = list(retrieved_docs)[:max_context_docs]

    for idx, row in enumerate(doc_rows, start=1):
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        doc_text = str(row.get("text", "")).strip()
        pub_date = str(row.get("publication_date", "Unknown")).strip()
        doc_id = str(row.get("docno", "")).strip()
        header = f"[Document {idx} | ID: {doc_id} | Date: {pub_date}]"
        context_chunks.append(f"{header}\n{doc_text}")

    context_str = "\n\n".join(context_chunks)

    prompt = (
        "You are an expert historical researcher answering questions based on 19th and early 20th century American newspapers.\n"
        "Read the following historical passages carefully and answer the question factually.\n"
        "Base your answer strictly on the provided documents. If the documents do not contain enough information, state so clearly.\n\n"
        f"--- CONTEXT PASSAGES ---\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return prompt


def generate_answer(query: str, reranked_docs: Any, llm, max_context_docs: int = 3) -> str:
    """Generate a synthesized factual answer using the top reranked passages and the LLM."""
    is_empty = getattr(reranked_docs, "empty", False) or len(reranked_docs) == 0
    if is_empty:
        return "No relevant historical documents were retrieved to answer the question."

    prompt = build_rag_prompt(query, reranked_docs, max_context_docs=max_context_docs)
    return llm.generate(prompt)
