"""Retrieval methods: BM25 and dense (sentence-transformer).

Provides a consistent interface for running retrieval with different methods
and models. Used by baseline evaluation and post-training comparison.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def bm25_retrieval(queries: list[str], corpus: list[str]) -> list[list[int]]:
    """Run BM25 retrieval. Returns ranked passage indices per query."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    rankings = []
    for query in queries:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1].tolist()
        rankings.append(ranked_indices)

    return rankings


def dense_retrieval(
    queries: list[str],
    corpus: list[str],
    model_name: str = "all-mpnet-base-v2",
) -> list[list[int]]:
    """Run dense retrieval with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Encoding {len(corpus)} passages...")
    t0 = time.time()
    corpus_emb = model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"  Corpus encoded in {time.time() - t0:.1f}s")

    logger.info(f"Encoding {len(queries)} queries...")
    t0 = time.time()
    query_emb = model.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"  Queries encoded in {time.time() - t0:.1f}s")

    similarities = cos_sim(query_emb, corpus_emb).numpy()

    rankings = []
    for i in range(len(queries)):
        ranked_indices = np.argsort(similarities[i])[::-1].tolist()
        rankings.append(ranked_indices)

    return rankings
