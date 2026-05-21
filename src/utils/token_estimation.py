"""Token count estimation using tiktoken.

Uses GPT-4's cl100k_base encoding as an approximation for token counting.
Accurate for OpenAI models; within ~5-10% for other models on English text.
"""

import tiktoken

_enc = tiktoken.encoding_for_model("gpt-4")


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return len(_enc.encode(text))


def estimate_skillbook_injected_tokens(skills: list[dict]) -> int:
    """Estimate tokens for skillbook as injected into agent prompt.

    Replicates the format from wrap_skillbook_context():
    ### {id}\n\n{content}[\n\n**Why this helps:** {justification}]
    """
    parts = []
    for s in skills:
        part = f"### {s.get('id', '')}\n\n{s.get('content', '')}"
        if s.get("justification"):
            part += f"\n\n**Why this helps:** {s['justification']}"
        parts.append(part)
    return estimate_tokens("\n\n".join(parts))
