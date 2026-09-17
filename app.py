import os
import streamlit as st
from src.llm import LlamaWrapper
from src.rag import final_score, generate_answer
from src.retrieval import (
    DEFAULT_SQLITE_PATH,
    DEFAULT_TEXT_INDEX_PATH,
    index_exists,
    SQLiteBM25Retriever,
    create_sqlite_index,
)
from src.data import build_document_collection

st.set_page_config(
    page_title="Historical News QA — RAG System",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Historical News Question Answering")
st.caption("BM25 Retrieval + Heuristic & Neural Reranking + RAG Generation (Chronicling America 1800–1920)")


# ==========================================
# Caching in memoria: Modello & Retriever
# ==========================================
@st.cache_resource(show_spinner="Caricamento modello LLM in memoria...")
def load_model():
    """Inizializza e mantiene l'LLM in memoria RAM / GPU."""
    return LlamaWrapper()


@st.cache_resource(show_spinner="Caricamento indice BM25 dalla cache...")
def load_search_pipeline():
    """Carica il retriever BM25 (priorità: SQLite FTS5 Pure-Python, fallback: PyTerrier)."""
    if os.path.exists(DEFAULT_SQLITE_PATH):
        # Pure-Python BM25: istantaneo, zero Java, zero C++
        retriever = SQLiteBM25Retriever(DEFAULT_SQLITE_PATH)
        return retriever, "pure_python"
    elif os.path.exists(DEFAULT_TEXT_INDEX_PATH):
        # PyTerrier Index
        import pyterrier as pt
        from src.retrieval import load_index, get_retriever
        index = load_index(DEFAULT_TEXT_INDEX_PATH)
        bm25 = get_retriever(index, wmodel="BM25", top_k=100)
        get_doc_properties = pt.text.get_text(index, ["text", "publication_date"])
        return (bm25, get_doc_properties), "pyterrier"
    else:
        return None, "none"


# ==========================================
# Controllo Indice all'Avvio
# ==========================================
has_sqlite = os.path.exists(DEFAULT_SQLITE_PATH)
has_pyterrier = index_exists(DEFAULT_TEXT_INDEX_PATH)

if not (has_sqlite or has_pyterrier):
    st.warning("⚠️ Nessun indice trovato.")
    st.markdown(
        """
        ### Come ottenere l'indice (Zero-Java / Pure Python):
        1. Esegui lo script di export in **Google Colab** sul dataset completo per generare `historical_news.db`.
        2. Scarica e sposta il file `historical_news.db` in questa cartella di progetto.
        3. Ricarica questa pagina!
        """
    )

    st.markdown("---")
    st.subheader("Oppure costruiscilo in locale dai file JSON:")
    if st.button("🔨 Costruisci `historical_news.db` adesso in locale", type="primary"):
        with st.spinner("Creazione indice SQLite FTS5 in corso..."):
            try:
                if os.path.exists("document_collection.json"):
                    import json
                    with open("document_collection.json", "r", encoding="utf-8") as f:
                        docs = json.load(f)
                else:
                    docs = build_document_collection()

                create_sqlite_index(DEFAULT_SQLITE_PATH, docs)
                st.success("✅ Indice pure-Python creato con successo!")
                st.rerun()
            except Exception as e:
                st.error(f"Errore: {e}")
                st.info("Assicurati di aver scaricato train.json/test.json o di scaricare historical_news.db da Colab.")
    st.stop()


# ==========================================
# Interfaccia Principale
# ==========================================
retriever_obj, mode = load_search_pipeline()
llm = load_model()

with st.sidebar:
    st.header("⚙️ Stato Sistema")
    if mode == "pure_python":
        st.success("✅ BM25 Pure Python (SQLite FTS5)")
        st.caption("Zero Java • Zero C++ • Prestazioni istantanee")
    else:
        st.success("✅ BM25 PyTerrier")
    top_k = st.slider("Numero di documenti da recuperare (Top-K)", min_value=1, max_value=20, value=10)

query = st.text_input(
    "Inserisci la tua domanda storica in linguaggio naturale:",
    placeholder="Es: When did Lincoln deliver the Gettysburg address?"
)

if st.button("🔍 Cerca e Rispondi", type="primary") and query:
    with st.spinner("1/3 Retrieval lessicale BM25..."):
        if mode == "pure_python":
            bm25_results = retriever_obj.search(query, top_k=top_k)
        else:
            bm25, get_doc_properties = retriever_obj
            bm25_results = bm25.search(query).head(top_k)
            bm25_results["text"] = get_doc_properties.transform(bm25_results)["text"]

    is_empty = getattr(bm25_results, "empty", False) or len(bm25_results) == 0
    if is_empty:
        st.warning("Nessun documento trovato per la query specificata.")
    else:
        with st.spinner("2/3 Re-ranking avanzato (Metadati temporali + NER)..."):
            # Garantisce campo query per scoring temporale
            if hasattr(bm25_results, "assign"):
                bm25_results["query"] = query
                bm25_results["llm_score"] = bm25_results.apply(
                    lambda row: final_score(row, llm), axis=1
                )
                reranked = bm25_results.sort_values("llm_score", ascending=False).reset_index(drop=True)
            else:
                for r in bm25_results:
                    r["query"] = query
                    r["llm_score"] = final_score(r, llm)
                reranked = sorted(bm25_results, key=lambda x: x["llm_score"], reverse=True)

        with st.spinner("3/3 Generazione della risposta con LLM..."):
            answer = generate_answer(query, reranked, llm)

        st.subheader("💡 Risposta Generata")
        st.info(answer)

        st.subheader("📊 Confronto Ranking (BM25 vs Reranked)")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Baseline: **BM25**")
            if hasattr(bm25_results, "head"):
                st.dataframe(bm25_results[["docno", "score", "publication_date"]].head(top_k), use_container_width=True)
            else:
                st.write(bm25_results[:top_k])

        with col2:
            st.markdown("#### Pipeline Avanzata: **Reranked**")
            if hasattr(reranked, "head"):
                st.dataframe(reranked[["docno", "llm_score", "publication_date"]].head(top_k), use_container_width=True)
            else:
                st.write(reranked[:top_k])

        with st.expander("📄 Visualizza i testi dei passaggi storici recuperati"):
            docs_iterable = reranked.head(3).iterrows() if hasattr(reranked, "iterrows") else enumerate(reranked[:3])
            for idx, item in enumerate(docs_iterable, start=1):
                row = item[1] if hasattr(reranked, "iterrows") else item
                if hasattr(row, "to_dict"):
                    row = row.to_dict()
                st.markdown(f"**Documento {idx} — ID `{row.get('docno', '')}` (Data: {row.get('publication_date', 'N/D')})**")
                st.write(row.get("text", ""))
                st.divider()