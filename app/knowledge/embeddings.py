from __future__ import annotations


def embed_texts_sentence_transformers(
    texts: list[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> tuple[list[list[float]], int]:
    """Generate dense embeddings using sentence-transformers."""
    if not texts:
        return [], 0

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True)
    embeddings: list[list[float]] = [list(map(float, row)) for row in vectors]
    dim = len(embeddings[0]) if embeddings else 0
    return embeddings, dim
