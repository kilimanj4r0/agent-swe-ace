# prompts/outputs.py
"""Custom output models for SWE-bench optimized skillbook learning."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from ace_next.core.outputs import ExtractedLearning


class AntiPattern(BaseModel):
    """A behavior pattern that led to failure."""

    pattern: str = Field(..., description="The anti-pattern behavior")
    why_harmful: str = Field(..., description="Why this pattern causes problems")
    atomicity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="Evidence from execution showing this pattern"
    )


class Discovery(BaseModel):
    """A verified factual finding from the trajectory."""

    finding: str = Field(..., description="The verified factual discovery")
    atomicity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How atomic/focused this learning is"
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
        default=0.0, ge=0.0, le=1.0, description="How atomic/focused this learning is"
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


class SWEReflectorOutput(BaseModel):
    """Output from the SWE-optimized Reflector role.

    Key difference from default ReflectorOutput:
    - Separates anti_patterns, discoveries, and unvalidated_hypotheses
    - Anti-patterns are warnings about what NOT to do
    - Discoveries are verified facts (file locations, error messages)
    - Unvalidated hypotheses are claims that need verification
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Overall reasoning about the outcome")
    error_identification: str = Field(
        default="", description="Description of what went wrong (if applicable)"
    )
    error_location: str = Field(
        default="N/A", description="Exact step where error occurred or 'N/A'"
    )
    root_cause_analysis: str = Field(
        default="", description="Analysis of why errors occurred"
    )
    correct_approach: str = Field(
        ..., description="What the correct approach should be"
    )

    # NEW: Separate learning types instead of generic extracted_learnings
    anti_patterns: List[AntiPattern] = Field(
        default_factory=list,
        description="Behaviors that led to failure - WARNINGS for future agents"
    )
    discoveries: List[Discovery] = Field(
        default_factory=list,
        description="Verified factual findings (file locations, error messages, etc)"
    )
    unvalidated_hypotheses: List[UnvalidatedHypothesis] = Field(
        default_factory=list,
        description="Agent claims that were NOT tested/verified"
    )

    key_insight: str = Field(
        ..., description="The main lesson learned from this iteration"
    )
    confidence_in_analysis: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in the analysis"
    )
    skill_tags: List[SkillTag] = Field(
        default_factory=list, description="Classifications of strategy effectiveness"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )

    # Compatibility: Provide extracted_learnings for SkillManager
    @property
    def extracted_learnings(self) -> List["ExtractedLearning"]:
        """Convert to ExtractedLearning format for compatibility with SkillManager.

        SkillManager expects ReflectorOutput with extracted_learnings,
        so we convert anti_patterns, discoveries, and unvalidated_hypotheses
        to that format.
        """
        learnings = []

        # Convert anti-patterns
        for ap in self.anti_patterns:
            learnings.append(ExtractedLearning(
                learning=f"[ANTI-PATTERN] {ap.pattern}",
                atomicity_score=ap.atomicity_score,
                evidence=ap.evidence,
                justification=ap.why_harmful,
            ))

        # Convert discoveries
        for d in self.discoveries:
            learnings.append(ExtractedLearning(
                learning=f"[DISCOVERY] {d.finding}",
                atomicity_score=d.atomicity_score,
                evidence=d.evidence,
            ))

        # Convert unvalidated hypotheses
        for uh in self.unvalidated_hypotheses:
            learnings.append(ExtractedLearning(
                learning=f"[HYPOTHESIS] {uh.hypothesis}",
                atomicity_score=uh.atomicity_score,
                evidence=uh.evidence,
                justification=uh.why_unvalidated,
            ))

        return learnings

    # Compatibility: Convert to list of dicts for code expecting extracted_learnings
    def get_all_learnings_as_dicts(self) -> List[Dict[str, Any]]:
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
