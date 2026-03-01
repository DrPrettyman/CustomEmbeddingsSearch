"""Pydantic models for ESG embedding training data."""

from pydantic import BaseModel, Field


class QueryPassagePair(BaseModel):
    """A single (query, passage) pair for training or evaluation."""

    query: str = Field(..., min_length=1, description="Search query an ESG analyst would use")
    passage: str = Field(..., min_length=1, description="Relevant passage from an ESG document")
    source: str = Field(
        ..., description="Data source: esgbench, cdp, gri, tcfd, or synthetic"
    )
    topic: str = Field(
        default="unknown",
        description="ESG topic category: environmental, social, governance, or unknown",
    )
    doc_id: str = Field(default="", description="Source document identifier")
    metadata: dict = Field(default_factory=dict, description="Additional source-specific metadata")


class EvalQuery(BaseModel):
    """An evaluation query with known relevant passage IDs."""

    query_id: str
    query: str
    relevant_passage_ids: list[str] = Field(
        ..., min_length=1, description="IDs of passages relevant to this query"
    )
    topic: str = Field(default="unknown")
    difficulty: str = Field(
        default="unknown",
        description="Query difficulty: keyword_overlap, paraphrase, conceptual, or unknown",
    )


class CorpusPassage(BaseModel):
    """A passage in the evaluation corpus."""

    passage_id: str
    passage: str
    source: str
    topic: str = Field(default="unknown")
    doc_id: str = Field(default="")
