import json
import os
import re
import unicodedata
import string


def load_list_or_empty(path: str) -> list:
    """Safely load a JSON list file, returning an empty list if missing or malformed."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def project(records: list) -> list:
    """Project record dictionaries to standard fields: para_id, context, raw_ocr, publication_date."""
    out = []
    for r in records:
        out.append({
            "para_id": r.get("para_id", ""),
            "context": r.get("context", ""),
            "raw_ocr": r.get("raw_ocr", ""),
            "publication_date": r.get("publication_date", "")
        })
    return out


def build_document_collection(
    input_files: list = None,
    output_file: str = "document_collection.json"
) -> list:
    """Aggregate train, validation, and test splits into a deduplicated document collection."""
    if input_files is None:
        input_files = ["train.json", "validation.json", "test.json"]

    all_recs = []
    for p in input_files:
        recs = load_list_or_empty(p)
        all_recs.extend(project(recs))

    # Deduplicate by para_id keeping the first one seen
    uniq = {}
    for rec in all_recs:
        pid = rec.get("para_id", "")
        if pid and pid not in uniq:
            uniq[pid] = rec

    result = list(uniq.values())

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def clean_question(text: str) -> str:
    """Normalize query text: NFKC normalization, strip punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_queries(
    input_file: str = "test.json",
    output_file: str = "test_queries.json",
    max_queries: int = 10000
) -> list:
    """Extract, clean, sort, and export queries from a dataset split."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [
        {
            "query_id": item.get("query_id", ""),
            "question": clean_question(item.get("question", "")),
        }
        for item in data
    ]

    queries = sorted(
        queries,
        key=lambda x: int(x["query_id"]) if str(x["query_id"]).isdigit() else x["query_id"]
    )

    if max_queries is not None:
        queries = queries[:max_queries]

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)

    return queries


def prepare_qrels_and_answers(
    input_file: str = "test.json",
    qrels_file: str = "test_qrels.json",
    answers_file: str = "test_query_answers.json"
) -> tuple:
    """Generate qrels and query answers data structures for evaluation."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    qrels = [
        {
            "query_id": item.get("query_id", ""),
            "iteration": 0,
            "para_id": item.get("para_id", ""),
            "relevance": 1
        }
        for item in data
    ]

    query_answers = [
        {
            "query_id": item.get("query_id", ""),
            "iteration": 0,
            "para_id": item.get("para_id", ""),
            "relevance": 1,
            "answer": item.get("answer", ""),
            "org_answer": item.get("org_answer", "")
        }
        for item in data
    ]

    if qrels_file:
        with open(qrels_file, "w", encoding="utf-8") as f:
            json.dump(qrels, f, ensure_ascii=False, indent=2)

    if answers_file:
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(query_answers, f, ensure_ascii=False, indent=2)

    return qrels, query_answers


def prepare_docs_for_indexing(data_list: list):
    """Generator formatting documents for PyTerrier IterDictIndexer."""
    for doc in data_list:
        yield {
            "docno": str(doc.get("para_id", "")),
            "text": doc.get("context", ""),
            "publication_date": str(doc.get("publication_date", "")),
            "raw_ocr": doc.get("raw_ocr", "")
        }


def format_topics_and_qrels(
    queries_path: str = "test_queries.json",
    qrels_path: str = "test_qrels.json"
) -> tuple:
    """Load and format topics and qrels into PyTerrier-compatible pandas DataFrames."""
    import pandas as pd
    with open(queries_path, "r", encoding="utf-8") as f:
        topics = pd.DataFrame(json.load(f))
    topics = topics.rename(columns={"query_id": "qid", "question": "query"})
    topics["qid"] = topics["qid"].astype(str)

    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = pd.DataFrame(json.load(f))
    qrels = qrels.rename(columns={"query_id": "qid", "para_id": "docno", "relevance": "label"})
    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["label"] = qrels["label"].astype(int)

    return topics, qrels
