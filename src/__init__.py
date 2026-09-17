"""Historical News Question Answering Package.

Modules:
- data: Dataset loading, cleaning, query processing, and PyTerrier formatting.
- retrieval: PyTerrier indexing, BM25 / TF-IDF retrievers, and metadata transformers.
- custom_reranker: Temporal and NER heuristic re-ranking (Phase II Experiment E2).
- neural_reranker: Sentence-transformers Cross-Encoder re-ranking (Phase II Experiment E1).
- llm: Language model wrappers for generation and local inference.
- rag: RAG prompt construction, candidate scoring, and answer generation.
"""

from src.data import (
    clean_question,
    load_list_or_empty,
    project,
    build_document_collection,
    prepare_queries,
    prepare_qrels_and_answers,
    prepare_docs_for_indexing,
    format_topics_and_qrels,
)

from src.custom_reranker import (
    extract_year,
    doc_year,
    temporal_score,
    ner_score,
    final_score as custom_final_score,
    build_custom_pipeline,
)

from src.neural_reranker import (
    get_cross_encoder,
    cross_encoder_score_single_query,
    build_neural_pipeline,
)

from src.retrieval import (
    init_pyterrier,
    build_index,
    load_index,
    index_exists,
    get_retriever,
    bm25_100_text,
    tfidf_text,
    bm25_100_raw_ocr,
    tfidf_raw_ocr,
    get_text,
    get_doc_properties,
    SQLiteBM25Retriever,
    create_sqlite_index,
    DEFAULT_SQLITE_PATH,
    DEFAULT_TEXT_INDEX_PATH,
)

from src.llm import LlamaWrapper
from src.rag import final_score, generate_answer, build_rag_prompt

__all__ = [
    "clean_question",
    "load_list_or_empty",
    "project",
    "build_document_collection",
    "prepare_queries",
    "prepare_qrels_and_answers",
    "prepare_docs_for_indexing",
    "format_topics_and_qrels",
    "extract_year",
    "doc_year",
    "temporal_score",
    "ner_score",
    "custom_final_score",
    "build_custom_pipeline",
    "get_cross_encoder",
    "cross_encoder_score_single_query",
    "build_neural_pipeline",
    "init_pyterrier",
    "build_index",
    "load_index",
    "index_exists",
    "get_retriever",
    "bm25_100_text",
    "tfidf_text",
    "bm25_100_raw_ocr",
    "tfidf_raw_ocr",
    "get_text",
    "get_doc_properties",
    "SQLiteBM25Retriever",
    "create_sqlite_index",
    "DEFAULT_SQLITE_PATH",
    "DEFAULT_TEXT_INDEX_PATH",
    "LlamaWrapper",
    "final_score",
    "generate_answer",
    "build_rag_prompt",
]
