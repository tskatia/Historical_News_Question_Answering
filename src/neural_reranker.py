from typing import Optional

_cross_encoder_model = None


def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy load the sentence-transformers CrossEncoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder_model = CrossEncoder(model_name)
    return _cross_encoder_model


def cross_encoder_score_single_query(df, model=None):
    """Score query-document pairs in a DataFrame using a cross-encoder model."""
    if df.empty:
        return df

    if model is None:
        model = get_cross_encoder()

    pairs = [[row["query"], row["text"]] for _, row in df.iterrows()]
    scores = model.predict(pairs, batch_size=len(df), show_progress_bar=False)

    df = df.copy()
    df["score"] = scores
    return df


def build_neural_pipeline(
    retriever,
    index,
    top_k: int = 10,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
):
    """Construct PyTerrier E1 pipeline: BM25 % top_k >> get_text >> neural cross-encoder."""
    import pyterrier as pt
    if not pt.started():
        pt.init()

    model = get_cross_encoder(model_name)
    get_text = pt.text.get_text(index, "text")

    def neural_scorer(df):
        return cross_encoder_score_single_query(df, model=model)

    neural_model = pt.apply.by_query(neural_scorer)
    return retriever % top_k >> get_text >> neural_model
