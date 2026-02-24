"""Pydantic models for logic knowledge base representation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Constant(BaseModel):
    """Represents a constant in the logic knowledge base."""

    id: str
    type: Literal["person", "location", "time", "organization", "object", "unknown"] = "unknown"
    gloss: str


class Predicate(BaseModel):
    """Represents a predicate with its arity and description."""

    name: str
    arity: int = Field(..., ge=1, le=8)
    gloss: str


class FOLBlock(BaseModel):
    """First-Order Logic representation block."""

    facts: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    optional_rules: list[str] = Field(default_factory=list)
    hypothesis: str = Field(default="", description="FOL hypothesis for the question/claim")


class PrologBlock(BaseModel):
    """Prolog representation block."""

    facts: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    optional_rules: list[str] = Field(default_factory=list)
    hypothesis: str = Field(default="", description="Prolog hypothesis for the question/claim")


class LogicKB(BaseModel):
    """Complete logic knowledge base with symbols and representations."""

    constants: list[Constant] = Field(default_factory=list)
    predicates: list[Predicate] = Field(default_factory=list)
    fol: FOLBlock
    prolog: PrologBlock


class LogicKBList(BaseModel):
    """Container for multiple logic knowledge bases."""

    items: list[LogicKB] = Field(default_factory=list)
