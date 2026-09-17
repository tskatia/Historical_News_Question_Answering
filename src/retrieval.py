import os
import re
import shutil
import sqlite3
from typing import Optional, List, Any

# Standard paths
DEFAULT_TEXT_INDEX_PATH = "./text-index"
DEFAULT_RAW_OCR_INDEX_PATH = "./raw_ocr-index"
DEFAULT_SQLITE_PATH = "historical_news.db"

_pt_initialized = False


# =========================================================================
# 1. Pure-Python BM25 Retriever (SQLite FTS5 — Zero Java, Zero C++ tools)
# =========================================================================

def create_sqlite_index(
    db_path: str = DEFAULT_SQLITE_PATH,
    docs = None,
    overwrite: bool = True
) -> str:
    """Build a pure-Python SQLite FTS5 BM25 index from a document collection or list."""
    if os.path.exists(db_path):
        if overwrite:
            os.remove(db_path)
        else:
            return db_path

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create FTS5 virtual table with BM25 ranking capability
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
            docno UNINDEXED,
            publication_date UNINDEXED,
            text,
            tokenize = 'porter unicode61'
        );
    """)

    if docs is not None:
        batch = []
        for doc in docs:
            docno = str(doc.get("docno") or doc.get("para_id") or "")
            pub_date = str(doc.get("publication_date") or "")
            text = str(doc.get("text") or doc.get("context") or "")
            batch.append((docno, pub_date, text))

            if len(batch) >= 5000:
                cursor.executemany(
                    "INSERT INTO articles_fts(docno, publication_date, text) VALUES (?, ?, ?);",
                    batch
                )
                batch = []

        if batch:
            cursor.executemany(
                "INSERT INTO articles_fts(docno, publication_date, text) VALUES (?, ?, ?);",
                batch
            )

    conn.commit()
    conn.close()
    return db_path


class SQLiteBM25Retriever:
    """Pure-Python BM25 Retriever powered by SQLite FTS5 (standard library, no Java)."""

    def __init__(self, db_path: str = DEFAULT_SQLITE_PATH):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database '{db_path}' not found.")

    def search(self, query: str, top_k: int = 10) -> Any:
        """Search query using SQLite FTS5 BM25 ranking.
        Returns a pandas DataFrame (if pandas installed) or list of dicts.
        """
        cleaned = re.sub(r"[^\w\s]", " ", query).strip()
        tokens = [t for t in cleaned.split() if t]
        if not tokens:
            records = []
        else:
            fts_query = " OR ".join(tokens)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # In SQLite FTS5, bm25() returns negative/lower scores for better matches.
            # We negate it so higher score = better rank.
            sql = """
                SELECT docno, (-1.0 * bm25(articles_fts)) AS score, text, publication_date
                FROM articles_fts
                WHERE articles_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """
            try:
                cursor.execute(sql, (fts_query, top_k))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                rows = []
            finally:
                conn.close()

            records = [
                {
                    "docno": str(r[0]),
                    "score": float(r[1]),
                    "text": str(r[2]),
                    "publication_date": str(r[3]),
                    "query": query
                }
                for r in rows
            ]

        try:
            import pandas as pd
            return pd.DataFrame(records)
        except ImportError:
            return records


# =========================================================================
# 2. PyTerrier Indexing & Retrievers (Java-based)
# =========================================================================

def init_pyterrier():
    """Ensure PyTerrier is initialized."""
    global _pt_initialized
    import pyterrier as pt
    if not pt.started():
        pt.init()
    _pt_initialized = True
    return pt


def build_index(
    index_path: str = DEFAULT_TEXT_INDEX_PATH,
    docs_generator = None,
    meta: Optional[dict] = None,
    text_attrs: Optional[List[str]] = None,
    overwrite: bool = False
):
    """Build a PyTerrier index from a document generator."""
    pt = init_pyterrier()

    if meta is None:
        meta = {"publication_date": 20, "text": 4096, "docno": 40}
    if text_attrs is None:
        text_attrs = ["text"]

    if os.path.exists(index_path):
        if overwrite:
            shutil.rmtree(index_path)
        else:
            return pt.IndexFactory.of(index_path)

    indexer = pt.IterDictIndexer(
        index_path,
        meta=meta,
        text_attrs=text_attrs
    )
    indexref = indexer.index(docs_generator)
    return pt.IndexFactory.of(indexref)


def index_exists(index_path: Optional[str] = None) -> bool:
    """Check if either a SQLite FTS5 database or a PyTerrier index directory exists."""
    if index_path is None:
        return os.path.exists(DEFAULT_SQLITE_PATH) or index_exists(DEFAULT_TEXT_INDEX_PATH)

    if not os.path.exists(index_path):
        return False

    if os.path.isfile(index_path):
        return os.path.getsize(index_path) > 0

    properties_file = os.path.join(index_path, "data.properties")
    return os.path.exists(properties_file) or len(os.listdir(index_path)) > 0


def load_index(index_path: str = DEFAULT_TEXT_INDEX_PATH):
    """Load an existing PyTerrier index by path."""
    pt = init_pyterrier()
    if not index_exists(index_path):
        raise FileNotFoundError(
            f"Index path '{index_path}' does not exist or is empty."
        )
    return pt.IndexFactory.of(index_path)


def get_retriever(index_or_ref, wmodel: str = "BM25", top_k: Optional[int] = 100):
    """Create a PyTerrier retriever with the given weighting model and optional top-k cutoff."""
    pt = init_pyterrier()
    retriever = pt.terrier.Retriever(index_or_ref, wmodel=wmodel)
    if top_k is not None:
        retriever = retriever % top_k
    return retriever


class _LazyRetrieverProxy:
    """Proxy object that delays index loading until the retriever is actually invoked."""
    def __init__(self, index_path: str, wmodel: str = "BM25", top_k: Optional[int] = 100):
        self.index_path = index_path
        self.wmodel = wmodel
        self.top_k = top_k
        self._retriever = None

    def _get_retriever(self):
        if self._retriever is None:
            idx = load_index(self.index_path)
            self._retriever = get_retriever(idx, wmodel=self.wmodel, top_k=self.top_k)
        return self._retriever

    def search(self, *args, **kwargs):
        return self._get_retriever().search(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self._get_retriever().transform(*args, **kwargs)

    def __mod__(self, k):
        return self._get_retriever() % k

    def __rshift__(self, other):
        return self._get_retriever() >> other

    def __getattr__(self, name):
        return getattr(self._get_retriever(), name)


class _LazyPropertyTransformerProxy:
    """Proxy for document text/metadata retrieval transformers."""
    def __init__(self, index_path: str, fields):
        self.index_path = index_path
        self.fields = fields
        self._transformer = None

    def _get_transformer(self):
        if self._transformer is None:
            pt = init_pyterrier()
            idx = load_index(self.index_path)
            self._transformer = pt.text.get_text(idx, self.fields)
        return self._transformer

    def transform(self, *args, **kwargs):
        return self._get_transformer().transform(*args, **kwargs)

    def __rshift__(self, other):
        return self._get_transformer() >> other

    def __getattr__(self, name):
        return getattr(self._get_transformer(), name)


# Standard lazy objects exported for backward compatibility
bm25_100_text = _LazyRetrieverProxy(DEFAULT_TEXT_INDEX_PATH, wmodel="BM25", top_k=100)
tfidf_text = _LazyRetrieverProxy(DEFAULT_TEXT_INDEX_PATH, wmodel="TF_IDF", top_k=None)
bm25_100_raw_ocr = _LazyRetrieverProxy(DEFAULT_RAW_OCR_INDEX_PATH, wmodel="BM25", top_k=100)
tfidf_raw_ocr = _LazyRetrieverProxy(DEFAULT_RAW_OCR_INDEX_PATH, wmodel="TF_IDF", top_k=None)

get_text = _LazyPropertyTransformerProxy(DEFAULT_TEXT_INDEX_PATH, "text")
get_doc_properties = _LazyPropertyTransformerProxy(DEFAULT_TEXT_INDEX_PATH, ["text", "publication_date"])
