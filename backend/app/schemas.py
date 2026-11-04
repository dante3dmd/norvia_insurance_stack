from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Citation(BaseModel):
    """Structured metadata for a retrieved chunk and its relevance score."""

    document_id: str
    chunk: str
    page: int | None = None
    score: float = Field(..., ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    """Request body for the RAG chat endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1)
    document_ids: list[str] | None = None
    k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="before")
    @classmethod
    def accept_max_fragments_alias(cls, values):
        if isinstance(values, dict) and "max_fragments" in values and "k" not in values:
            values = dict(values)
            values["k"] = values["max_fragments"]
        return values

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("question cannot be empty")
        return clean


class ChatResponse(BaseModel):
    """Structured response that the LLM must validate before being returned to the client."""

    answer: str
    citations: list[Citation]
    has_sufficient_evidence: bool
