"""Dataset builder: merge, deduplicate, quality filter, and split ESG training data.

Combines pairs from all sources (ESGBench, CDP, synthetic) into a unified
training dataset with train/eval splits.
"""

import hashlib
import json
import logging
import random
from pathlib import Path

from src.data.schemas import CorpusPassage, EvalQuery, QueryPassagePair

logger = logging.getLogger(__name__)

# Quality filter thresholds
MIN_QUERY_CHARS = 15
MAX_QUERY_CHARS = 300
MIN_PASSAGE_CHARS = 50
MAX_PASSAGE_CHARS = 4000  # CDP answers can be long; truncate rather than discard
EVAL_FRACTION = 0.15  # 15% held out for evaluation


def deduplicate_pairs(pairs: list[QueryPassagePair]) -> list[QueryPassagePair]:
    """Remove duplicate (query, passage) pairs.

    Deduplicates on the (query, passage) combination. The same query with
    different passages is kept — this is valid for CDP questions that appear
    in every company's response with different answers.
    """
    seen = set()
    unique = []

    for pair in pairs:
        # Normalize for comparison
        norm_passage = " ".join(pair.passage.lower().split())
        norm_query = " ".join(pair.query.lower().split())

        # Deduplicate on the (query, passage) combination
        pair_key = (norm_query, hashlib.md5(norm_passage.encode()).hexdigest())
        if pair_key in seen:
            continue

        seen.add(pair_key)
        unique.append(pair)

    removed = len(pairs) - len(unique)
    if removed > 0:
        logger.info(f"Deduplication: removed {removed} pairs ({len(unique)} remaining)")
    return unique


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Find the last sentence-ending punctuation before the limit
    for end_char in [". ", ".\n", "? ", "!\n"]:
        last_idx = truncated.rfind(end_char)
        if last_idx > max_chars // 2:  # Don't cut more than half
            return truncated[: last_idx + 1].strip()
    # Fallback: cut at last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space].strip()
    return truncated.strip()


def quality_filter(pairs: list[QueryPassagePair]) -> list[QueryPassagePair]:
    """Apply quality filters. Long passages are truncated rather than discarded."""
    filtered = []
    reasons = {"query_too_short": 0, "query_too_long": 0, "passage_too_short": 0,
               "passage_truncated": 0}

    for pair in pairs:
        if len(pair.query) < MIN_QUERY_CHARS:
            reasons["query_too_short"] += 1
            continue
        if len(pair.query) > MAX_QUERY_CHARS:
            reasons["query_too_long"] += 1
            continue
        if len(pair.passage) < MIN_PASSAGE_CHARS:
            reasons["passage_too_short"] += 1
            continue
        if len(pair.passage) > MAX_PASSAGE_CHARS:
            truncated = _truncate_at_sentence(pair.passage, MAX_PASSAGE_CHARS)
            if len(truncated) < MIN_PASSAGE_CHARS:
                reasons["passage_too_short"] += 1
                continue
            pair = pair.model_copy(update={"passage": truncated})
            reasons["passage_truncated"] += 1
        filtered.append(pair)

    removed = len(pairs) - len(filtered)
    if removed > 0:
        logger.info(f"Quality filter: removed {removed} pairs. Reasons: {reasons}")
    if reasons["passage_truncated"] > 0:
        logger.info(f"Quality filter: truncated {reasons['passage_truncated']} long passages")
    return filtered


def split_train_eval(
    pairs: list[QueryPassagePair],
    eval_fraction: float = EVAL_FRACTION,
    seed: int = 42,
    prefer_eval_sources: tuple[str, ...] = ("esgbench",),
) -> tuple[list[QueryPassagePair], list[QueryPassagePair]]:
    """Split pairs into training and evaluation sets.

    Prefers high-quality sources (ESGBench, real CDP Q&A) for the eval set,
    since these have real analyst questions rather than synthetic ones.
    """
    rng = random.Random(seed)

    # Separate preferred eval sources
    eval_preferred = [p for p in pairs if p.source in prefer_eval_sources]
    other = [p for p in pairs if p.source not in prefer_eval_sources]

    target_eval_size = max(int(len(pairs) * eval_fraction), 20)

    # Use all preferred sources for eval if they fit
    if len(eval_preferred) <= target_eval_size:
        eval_set = eval_preferred
        remaining_eval = target_eval_size - len(eval_set)
        # Fill remaining eval from other sources
        rng.shuffle(other)
        eval_set.extend(other[:remaining_eval])
        train_set = other[remaining_eval:]
    else:
        # Too many preferred — sample from them
        rng.shuffle(eval_preferred)
        eval_set = eval_preferred[:target_eval_size]
        train_set = eval_preferred[target_eval_size:] + other

    rng.shuffle(train_set)
    logger.info(f"Split: {len(train_set)} train, {len(eval_set)} eval")
    return train_set, eval_set


def build_eval_corpus(
    eval_pairs: list[QueryPassagePair],
    train_pairs: list[QueryPassagePair],
    max_corpus_size: int = 5000,
) -> tuple[list[EvalQuery], list[CorpusPassage]]:
    """Build evaluation corpus from eval pairs plus distractors from training data.

    The eval corpus contains:
    - All passages from eval pairs (the relevant passages)
    - A sample of passages from training pairs (distractors)
    """
    # Build corpus of unique passages
    corpus = {}
    qrels = {}  # query_id -> [passage_ids]

    # Add eval passages
    for i, pair in enumerate(eval_pairs):
        p_id = hashlib.md5(pair.passage.encode()).hexdigest()[:12]
        corpus[p_id] = CorpusPassage(
            passage_id=p_id,
            passage=pair.passage,
            source=pair.source,
            topic=pair.topic,
            doc_id=pair.doc_id,
        )
        q_id = f"eval_{i:04d}"
        if q_id not in qrels:
            qrels[q_id] = {"query": pair.query, "topic": pair.topic, "passage_ids": []}
        qrels[q_id]["passage_ids"].append(p_id)

    # Add distractor passages from training data
    rng = random.Random(42)
    train_passages = list({p.passage for p in train_pairs})
    rng.shuffle(train_passages)
    remaining_slots = max_corpus_size - len(corpus)

    for passage_text in train_passages[:remaining_slots]:
        p_id = hashlib.md5(passage_text.encode()).hexdigest()[:12]
        if p_id not in corpus:
            # Find the source info from the first matching training pair
            source_pair = next((p for p in train_pairs if p.passage == passage_text), None)
            corpus[p_id] = CorpusPassage(
                passage_id=p_id,
                passage=passage_text,
                source=source_pair.source if source_pair else "unknown",
                topic=source_pair.topic if source_pair else "unknown",
            )

    # Build EvalQuery objects
    eval_queries = []
    for q_id, info in qrels.items():
        eval_queries.append(
            EvalQuery(
                query_id=q_id,
                query=info["query"],
                relevant_passage_ids=info["passage_ids"],
                topic=info["topic"],
            )
        )

    corpus_list = list(corpus.values())
    logger.info(
        f"Eval corpus: {len(eval_queries)} queries, {len(corpus_list)} passages "
        f"({len(corpus_list) - len(eval_pairs)} distractors)"
    )
    return eval_queries, corpus_list


def save_dataset(
    train_pairs: list[QueryPassagePair],
    eval_queries: list[EvalQuery],
    eval_corpus: list[CorpusPassage],
    output_dir: Path,
) -> None:
    """Save the constructed dataset to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training pairs
    train_path = output_dir / "train_pairs.jsonl"
    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(pair.model_dump_json() + "\n")
    logger.info(f"Saved {len(train_pairs)} training pairs to {train_path}")

    # Eval queries
    eval_queries_path = output_dir / "eval_queries.jsonl"
    with open(eval_queries_path, "w") as f:
        for query in eval_queries:
            f.write(query.model_dump_json() + "\n")
    logger.info(f"Saved {len(eval_queries)} eval queries to {eval_queries_path}")

    # Eval corpus
    eval_corpus_path = output_dir / "eval_corpus.jsonl"
    with open(eval_corpus_path, "w") as f:
        for passage in eval_corpus:
            f.write(passage.model_dump_json() + "\n")
    logger.info(f"Saved {len(eval_corpus)} eval corpus passages to {eval_corpus_path}")

    # Query relevance judgments (for InformationRetrievalEvaluator)
    qrels = {}
    for query in eval_queries:
        qrels[query.query_id] = {pid: 1 for pid in query.relevant_passage_ids}
    qrels_path = output_dir / "eval_qrels.json"
    with open(qrels_path, "w") as f:
        json.dump(qrels, f, indent=2)
    logger.info(f"Saved qrels to {qrels_path}")

    # Dataset stats
    stats = {
        "train_pairs": len(train_pairs),
        "eval_queries": len(eval_queries),
        "eval_corpus": len(eval_corpus),
        "sources": {},
        "topics": {},
    }
    for pair in train_pairs:
        stats["sources"][pair.source] = stats["sources"].get(pair.source, 0) + 1
        stats["topics"][pair.topic] = stats["topics"].get(pair.topic, 0) + 1

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset stats: {stats}")
