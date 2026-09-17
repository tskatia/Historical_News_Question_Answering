import re
from typing import Optional, List

_nlp = None


def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """Lazy load the spaCy NLP model, returning None if spaCy is not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load(model_name)
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
                _nlp = spacy.load(model_name)
        except (ImportError, Exception):
            return None
    return _nlp


def extract_year(text: str) -> Optional[int]:
    """Extract a 4-digit year in range 1800-1920 from text using regex."""
    if not isinstance(text, str):
        return None
    match = re.search(r"\b(18\d{2}|19[0-2]\d)\b", text)
    if match is not None:
        return int(match.group(1))
    return None


def doc_year(publ_date_str: str) -> Optional[int]:
    """Extract the 4-digit year component from document publication date metadata."""
    if publ_date_str is None:
        return None

    s = str(publ_date_str).strip()
    if len(s) < 4:
        return None

    try:
        return int(s[:4])
    except ValueError:
        return None


def temporal_score(row) -> float:
    """Adjust BM25 score based on temporal proximity between query year and document year."""
    q_year = extract_year(row.get("query", ""))
    d_year = doc_year(row.get("publication_date", None))
    score = float(row.get("score", 0.0))

    if q_year is not None and d_year is not None:
        diff = abs(q_year - d_year)
        if diff == 0:
            score *= 1.5
        elif diff <= 2:
            score *= 1.2
        elif diff <= 5:
            score *= 1.1

    return score


def entities_match(query_entities: List[str], doc_entities: List[str]) -> bool:
    """Check if any named entity in the query exists in the document text."""
    for e in query_entities:
        if e in doc_entities:
            return True
    return False


def ner_score(row, nlp=None) -> float:
    """Penalize document score (0.5x) if named entities in query are missing in document.
    Uses spaCy if available, otherwise falls back to pure-Python proper noun matching.
    """
    if nlp is None:
        nlp = get_spacy_nlp()

    query_text = row.get("query", "")
    doc_text = row.get("text", "")

    if nlp is not None:
        query_nlp = nlp(query_text)
        query_entities = [e.text for e in query_nlp.ents]
        doc_nlp = nlp(doc_text)
        doc_entities = [e.text for e in doc_nlp.ents]
    else:
        # Pure-Python fallback: match capitalized entity words (e.g. Lincoln, Gettysburg)
        query_entities = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", query_text)
        doc_entities = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", doc_text))

    # If the query has no named entities, no penalty is applied
    if not query_entities:
        return float(row.get("score", 0.0))

    score = float(row.get("score", 0.0))
    if not entities_match(query_entities, doc_entities):
        score *= 0.5

    return score


def final_score(row, nlp=None) -> float:
    """Compute combined heuristic score for Phase II Experiment E2."""
    return temporal_score(row) + ner_score(row, nlp=nlp)


def build_custom_pipeline(retriever, index, top_k: int = 10, nlp=None):
    """Construct PyTerrier E2 pipeline: BM25 % top_k >> get_doc_properties >> custom score."""
    import pyterrier as pt
    if not pt.started():
        pt.init()

    get_doc_properties = pt.text.get_text(index, ["text", "publication_date"])

    def scoring_fn(row):
        return final_score(row, nlp=nlp)

    return retriever % top_k >> get_doc_properties >> pt.apply.doc_score(scoring_fn)
