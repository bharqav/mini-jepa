"""Typed decision-model interface (spec section 3)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["choice", "score", "noul"]


class Candidate(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class QuestionRequest(BaseModel):
    id: str = Field(..., min_length=1)
    type: QuestionType
    question: str = Field(..., min_length=1)
    candidates: list[Candidate] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # score scale, e.g. [0, 1]; normalized score = expected_level / (n_levels - 1)
    scale: list[float] | None = None

    @model_validator(mode="after")
    def _check(self):
        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate ids")
        if self.type == "choice" and len(self.candidates) < 2:
            raise ValueError("choice requires >= 2 candidates")
        if self.type == "noul" and len(self.candidates) != 2:
            raise ValueError("noul requires exactly 2 candidates ([no, yes] semantics)")
        if self.type == "score" and len(self.candidates) < 2:
            raise ValueError("score requires >= 2 ordered level candidates")
        return self


class InferenceSettings(BaseModel):
    max_state_tokens: int = 512
    max_candidate_tokens: int = 64
    temperature: float = 1.0
    include_metadata: bool = False
    # bounded internal chunking; None = one pass over all questions
    max_batch: int | None = None


class DecisionRequest(BaseModel):
    state: str = Field(..., min_length=1)
    questions: list[QuestionRequest] = Field(..., min_length=1)
    settings: InferenceSettings = Field(default_factory=InferenceSettings)

    @model_validator(mode="after")
    def _check(self):
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate question ids")
        return self


class CandidateScore(BaseModel):
    id: str
    score: float  # probability (choice/noul) or level probability (score)


class QuestionResponse(BaseModel):
    id: str
    type: QuestionType
    scores: list[CandidateScore]
    selected: str | None = None
    confidence: float
    # score-only extras
    expected_level: float | None = None
    normalized_score: float | None = None


class DecisionResponse(BaseModel):
    results: list[QuestionResponse]
    model_version: str | None = None
    inference_metadata: dict[str, Any] | None = None


def validate_response(resp: DecisionResponse) -> None:
    """Raise if any probability vector is invalid. Never silently accept garbage."""
    for r in resp.results:
        if r.type in ("choice", "noul"):
            total = sum(s.score for s in r.scores)
            if not all(s.score >= 0 and s.score == s.score for s in r.scores):
                raise ValueError(f"question {r.id}: non-finite/negative probability")
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"question {r.id}: probabilities sum to {total}, expected 1")
