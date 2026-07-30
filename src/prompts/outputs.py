"""Custom output models for SWE-bench optimized skillbook learning."""

from typing import List

from ace.core.outputs import ExtractedLearning, ReflectorOutput
from pydantic import BaseModel, Field, model_validator


class AntiPattern(BaseModel):
    """A behavior pattern that led to failure."""

    pattern: str = Field(..., description="The anti-pattern behavior")
    why_harmful: str = Field(..., description="Why this pattern causes problems")
    atomicity_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="Evidence from execution showing this pattern"
    )


class Discovery(BaseModel):
    """A verified factual finding from the trajectory."""

    finding: str = Field(..., description="The verified factual discovery")
    atomicity_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="How this was verified"
    )


class UnvalidatedHypothesis(BaseModel):
    """An agent claim that was not tested or verified."""

    hypothesis: str = Field(..., description="The unvalidated claim")
    why_unvalidated: str = Field(
        ..., description="What test/verification was missing"
    )
    atomicity_score: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="Agent's reasoning without proof"
    )


class SkillTag(BaseModel):
    """Classification tag for a skill strategy (helpful/harmful/neutral)."""

    id: str = Field(..., description="The skill ID being tagged")
    tag: str = Field(
        ..., description="Classification: 'helpful', 'harmful', or 'neutral'"
    )
    justification: str = Field(
        default="", description="Why this tag was assigned"
    )
    impact_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Impact of this skill on the outcome"
    )


class SWEReflectorOutput(ReflectorOutput):
    """SWE-optimized ReflectorOutput with typed learning categories.

    Extends ReflectorOutput to maintain isinstance() compatibility
    while adding SWE-specific fields.
    """

    # SWE-specific additional fields
    error_location: str = Field(
        default="N/A", description="Exact step where error occurred or 'N/A'"
    )
    confidence_in_analysis: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the analysis"
    )

    # Typed learning categories
    anti_patterns: List[AntiPattern] = Field(
        default_factory=list,
        description="Behaviors that led to failure - WARNINGS for future agents"
    )
    discoveries: List[Discovery] = Field(
        default_factory=list,
        description="Verified factual findings"
    )
    unvalidated_hypotheses: List[UnvalidatedHypothesis] = Field(
        default_factory=list,
        description="Agent claims that were NOT tested/verified"
    )

    @model_validator(mode="after")
    def _compute_extracted_learnings(self) -> "SWEReflectorOutput":
        """Compute extracted_learnings from typed categories after validation."""
        # Use object.__setattr__ to bypass Pydantic's frozen model protection
        learnings = []
        for ap in self.anti_patterns:
            learnings.append(ExtractedLearning(
                learning=f"[ANTI-PATTERN] {ap.pattern}",
                atomicity_score=ap.atomicity_score,
                evidence=ap.evidence,
                justification=ap.why_harmful,
            ))
        for d in self.discoveries:
            learnings.append(ExtractedLearning(
                learning=f"[DISCOVERY] {d.finding}",
                atomicity_score=d.atomicity_score,
                evidence=d.evidence,
            ))
        for uh in self.unvalidated_hypotheses:
            learnings.append(ExtractedLearning(
                learning=f"[HYPOTHESIS] {uh.hypothesis}",
                atomicity_score=uh.atomicity_score,
                evidence=uh.evidence,
                justification=uh.why_unvalidated,
            ))
        object.__setattr__(self, "extracted_learnings", learnings)
        return self

    def get_all_learnings_as_dicts(self) -> list:
        """Get all learnings as a flat list for compatibility."""
        learnings = []
        for ap in self.anti_patterns:
            learnings.append({
                "type": "anti_pattern",
                "learning": ap.pattern,
                "why_harmful": ap.why_harmful,
                "atomicity_score": ap.atomicity_score,
                "evidence": ap.evidence,
            })
        for d in self.discoveries:
            learnings.append({
                "type": "discovery",
                "learning": d.finding,
                "atomicity_score": d.atomicity_score,
                "evidence": d.evidence,
            })
        for uh in self.unvalidated_hypotheses:
            learnings.append({
                "type": "unvalidated_hypothesis",
                "learning": uh.hypothesis,
                "why_unvalidated": uh.why_unvalidated,
                "atomicity_score": uh.atomicity_score,
                "evidence": uh.evidence,
            })
        return learnings
