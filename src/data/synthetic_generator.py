"""Synthetic query generation for ESG passages using an LLM.

Given ESG text passages, generates diverse search queries that an analyst
would use to find each passage. This is the primary scaling mechanism
to reach 10-20K training pairs from a smaller set of real passages.

Supports multiple LLM backends:
- "claude-code": Uses the `claude -p` CLI tool (no API key needed)
- "anthropic": Uses the Anthropic Python SDK (requires ANTHROPIC_API_KEY)
- "openai": Uses the OpenAI Python SDK (requires OPENAI_API_KEY)
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from src.data.schemas import QueryPassagePair

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an ESG (Environmental, Social, Governance) research analyst.
Your task is to generate realistic search queries that an analyst would type
into a search engine to find the given passage from a sustainability report.

Generate exactly {n_queries} diverse queries for the passage. Vary the queries:
- One broad/general query (topic-level)
- One specific query (mentions particular metrics or standards)
- One paraphrase query (same concept, different words — no keyword overlap with the passage)

Rules:
- Each query should be 5-20 words
- Queries should sound natural — how a real analyst would search
- Do NOT copy phrases directly from the passage (except standard terms like "Scope 1")
- Include ESG-specific terminology where appropriate

Return ONLY a JSON array of strings, no other text. Example:
["What are Company X's direct greenhouse gas emissions?", "Scope 1 carbon footprint data 2024", "How much CO2 does the company emit from its own operations?"]"""

USER_PROMPT_TEMPLATE = """Generate {n_queries} search queries for this ESG passage:

---
{passage}
---

Company: {company}
Document type: {doc_type}

Return a JSON array of {n_queries} query strings."""


def _build_full_prompt(
    passage: str,
    company: str = "Unknown",
    doc_type: str = "sustainability report",
    n_queries: int = 3,
) -> str:
    """Build the full prompt combining system and user instructions."""
    system = SYSTEM_PROMPT.format(n_queries=n_queries)
    user = USER_PROMPT_TEMPLATE.format(
        n_queries=n_queries,
        passage=passage[:1500],
        company=company,
        doc_type=doc_type,
    )
    return f"{system}\n\n{user}"


def generate_queries_claude_code(
    passage: str,
    company: str = "Unknown",
    doc_type: str = "sustainability report",
    n_queries: int = 3,
) -> list[str]:
    """Generate synthetic queries using the `claude -p` CLI tool."""
    prompt = _build_full_prompt(passage, company, doc_type, n_queries)

    # Unset CLAUDECODE env var to allow nested invocation from within a Claude Code session
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    env["PATH"] = os.environ.get("PATH", "")
    env["HOME"] = os.environ.get("HOME", "")

    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr[:200]}")

    return _parse_query_response(result.stdout, n_queries)


def generate_queries_anthropic(
    passage: str,
    company: str = "Unknown",
    doc_type: str = "sustainability report",
    n_queries: int = 3,
    model: str = "claude-haiku-4-5-20251001",
) -> list[str]:
    """Generate synthetic queries using the Anthropic API.

    Requires ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT.format(n_queries=n_queries),
        messages=[
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    n_queries=n_queries,
                    passage=passage[:1500],
                    company=company,
                    doc_type=doc_type,
                ),
            }
        ],
    )

    return _parse_query_response(response.content[0].text, n_queries)


def generate_queries_openai(
    passage: str,
    company: str = "Unknown",
    doc_type: str = "sustainability report",
    n_queries: int = 3,
    model: str = "gpt-4o-mini",
) -> list[str]:
    """Generate synthetic queries using the OpenAI API.

    Requires OPENAI_API_KEY environment variable.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(n_queries=n_queries)},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    n_queries=n_queries,
                    passage=passage[:1500],
                    company=company,
                    doc_type=doc_type,
                ),
            },
        ],
    )

    return _parse_query_response(response.choices[0].message.content, n_queries)


def _parse_query_response(text: str, expected_count: int) -> list[str]:
    """Parse LLM response into a list of query strings."""
    text = text.strip()

    # Handle markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    try:
        queries = json.loads(text)
        if isinstance(queries, list):
            return [str(q).strip() for q in queries if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract quoted strings
    matches = re.findall(r'"([^"]+)"', text)
    if matches:
        return matches[:expected_count]

    # Last resort: split by newlines
    lines = [line.strip().lstrip("0123456789.-) ") for line in text.split("\n") if line.strip()]
    return [line for line in lines if len(line) > 10][:expected_count]


def generate_synthetic_pairs(
    passages: list[dict],
    backend: str = "claude-code",
    n_queries_per_passage: int = 3,
    rate_limit_delay: float = 0.5,
    max_passages: int | None = None,
    **kwargs,
) -> list[QueryPassagePair]:
    """Generate synthetic (query, passage) pairs for a list of passages.

    Args:
        passages: List of dicts with keys: text, source_file, company, doc_type, topic
        backend: "claude-code", "anthropic", or "openai"
        n_queries_per_passage: Number of queries to generate per passage
        rate_limit_delay: Seconds between API calls
        max_passages: Limit number of passages to process (for cost control)
        **kwargs: Additional args passed to the generation function

    Returns:
        List of QueryPassagePair objects
    """
    if backend == "claude-code":
        generate_fn = generate_queries_claude_code
    elif backend == "anthropic":
        generate_fn = generate_queries_anthropic
    elif backend == "openai":
        generate_fn = generate_queries_openai
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'claude-code', 'anthropic', or 'openai'")

    if max_passages:
        passages = passages[:max_passages]

    all_pairs = []
    errors = 0

    for i, passage_info in enumerate(passages):
        if i > 0 and rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        try:
            queries = generate_fn(
                passage=passage_info["text"],
                company=passage_info.get("company", "Unknown"),
                doc_type=passage_info.get("doc_type", "sustainability report"),
                n_queries=n_queries_per_passage,
                **kwargs,
            )

            for query in queries:
                pair = QueryPassagePair(
                    query=query,
                    passage=passage_info["text"],
                    source="synthetic",
                    topic=passage_info.get("topic", "unknown"),
                    doc_id=passage_info.get("source_file", ""),
                    metadata={
                        "company": passage_info.get("company", "Unknown"),
                        "generation_backend": backend,
                        "original_source": passage_info.get("original_source", ""),
                    },
                )
                all_pairs.append(pair)

        except Exception as e:
            errors += 1
            logger.warning(f"Failed to generate queries for passage {i}: {e}")
            if errors > 10:
                logger.error("Too many errors, stopping generation")
                break

        if (i + 1) % 50 == 0:
            logger.info(
                f"Generated queries for {i + 1}/{len(passages)} passages "
                f"({len(all_pairs)} pairs so far, {errors} errors)"
            )

    logger.info(
        f"Synthetic generation complete: {len(all_pairs)} pairs from "
        f"{len(passages)} passages ({errors} errors)"
    )
    return all_pairs
