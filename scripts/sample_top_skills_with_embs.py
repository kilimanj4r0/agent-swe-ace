"""
Retrieve top-k skills from a skillbook using embedding cosine similarity.

Deterministic, non-stochastic replacement for the LLM-based sample_top_skills_v2.py.

Pipeline:
  1. Embed all skills (section + content) — cached to disk.
  2. Embed the issue (title + body).
  3. Rank by cosine similarity, return top-k.

Usage:
    # From parquet (preferred):
    python sample_top_skills_with_embs.py \
        --skillbook final_skillbook_global.json \
        --parquet data.parquet \
        --instance-id astropy__astropy-12907 \
        -k 5

    # Manual mode:
    python sample_top_skills_with_embs.py \
        --skillbook final_skillbook_global.json \
        --repo django__django \
        --issue-title "Bug in migrations" \
        --issue-body "Description..." \
        -k 5
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Skill text representation
# ---------------------------------------------------------------------------


def _skill_text(skill: dict, *, include_section: bool = False) -> str:
    """Build a single text string from a skill dict for embedding."""
    parts = []
    if include_section:
        parts.append(skill.get("section", ""))
    parts.append(skill.get("content", ""))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _ensure_cache_dir() -> Path:
    """Create and return the .cache/ directory alongside this script."""
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR


def _skill_cache_path(model_name: str, skillbook_hash: str, include_section: bool) -> Path:
    """Return a deterministic cache path for pre-computed skill embeddings."""
    safe_model = model_name.replace("/", "__")
    suffix = "_wid" if include_section else "_noid"
    return _ensure_cache_dir() / f"skill_emb_{safe_model}_{skillbook_hash}{suffix}.npz"


def _query_cache_path(model_name: str, query_hash: str) -> Path:
    """Return a deterministic cache path for a query embedding."""
    safe_model = model_name.replace("/", "__")
    return _ensure_cache_dir() / f"query_emb_{safe_model}_{query_hash}.npy"


def _load_or_compute_skill_embeddings(
    skills: dict,
    skillbook_path: str,
    model_name: str,
    device: str,
    batch_size: int,
    include_section: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Return (skill_ids, embeddings_matrix) — loading from cache if available."""
    book_hash = hashlib.sha256(Path(skillbook_path).resolve().read_bytes()).hexdigest()[:12]
    cache = _skill_cache_path(model_name, book_hash, include_section)

    if cache.exists():
        print(f"Loading cached skill embeddings from {cache}", file=sys.stderr)
        data = np.load(cache, allow_pickle=False)
        return list(data["ids"]), data["embeddings"]

    print(f"Computing skill embeddings ({len(skills)} skills)...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": "bfloat16"},
    )

    skill_ids = list(skills.keys())
    texts = [_skill_text(skills[sid], include_section=include_section) for sid in skill_ids]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.savez(cache, ids=skill_ids, embeddings=embeddings)
    print(f"Cached skill embeddings to {cache}", file=sys.stderr)

    return skill_ids, embeddings


def _embed_query(text: str, model_name: str, device: str) -> np.ndarray:
    """Embed a single query string and return L2-normalised vector, with caching."""
    query_hash = hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()[:12]
    cache = _query_cache_path(model_name, query_hash)

    if cache.exists():
        return np.load(cache)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": "bfloat16"},
    )
    vec = model.encode(
        [text],
        normalize_embeddings=True,
    )
    vec = np.asarray(vec, dtype=np.float32)[0]
    np.save(cache, vec)
    print(f"Cached query embedding to {cache}", file=sys.stderr)
    return vec


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_top_k(
    skills: dict,
    issue_title: str,
    issue_body: str,
    k: int,
    *,
    repo: str = "",
    skillbook_path: str,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "cuda",
    batch_size: int = 32,
    include_identifiers: bool = False,
) -> list[tuple[str, float]]:
    """Return top-k (skill_id, similarity) pairs by cosine similarity."""
    skill_ids, skill_embs = _load_or_compute_skill_embeddings(
        skills, skillbook_path, model_name, device, batch_size,
        include_section=include_identifiers,
    )

    parts = []
    if include_identifiers and repo:
        parts.append(repo)
    parts.append(issue_title)
    parts.append(issue_body)
    issue_text = "\n".join(parts)
    query_vec = _embed_query(issue_text, model_name, device)

    # Cosine similarity (both vectors are L2-normalised)
    scores = skill_embs @ query_vec  # (num_skills,)

    top_indices = np.argsort(scores)[::-1][:k]
    return [(skill_ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_from_parquet(parquet_path: str, instance_id: str) -> tuple[str, str, str]:
    """Load issue data from a SWE-bench parquet file. Returns (repo, title, body)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    row = df[df["instance_id"] == instance_id]
    if row.empty:
        raise ValueError(f"instance_id '{instance_id}' not found in {parquet_path}")
    row = row.iloc[0]
    repo = str(row["repo"]).replace("/", "__")
    ps = str(row["problem_statement"])
    title = ps.split("\n", 1)[0]
    body = ps.split("\n", 1)[1] if "\n" in ps else ""
    return repo, title, body


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skillbook", required=True, help="Path to skillbook JSON")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet", help="Path to SWE-bench parquet (use with --instance-id)")
    src.add_argument("--repo", help="Repo name, e.g. django__django (manual mode)")

    ap.add_argument("--instance-id", help="Row id in parquet (required with --parquet)")
    ap.add_argument("--issue-title", help="Issue title (manual mode)")
    ap.add_argument("--issue-body", help="Issue description (manual mode)")
    ap.add_argument("-k", type=int, default=5, help="Number of skills to retrieve")
    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-4B",
        help="Sentence-transformers embedding model (default: Qwen/Qwen3-Embedding-4B)",
    )
    ap.add_argument("--device", default="cuda", help="Device for embedding model")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for skill embedding")
    ap.add_argument(
        "--include-identifiers",
        action="store_true",
        default=False,
        help="Include skill section in skill text and repo name in issue text",
    )
    args = ap.parse_args()

    if args.parquet:
        if not args.instance_id:
            ap.error("--instance-id is required when using --parquet")
        repo, issue_title, issue_body = _load_from_parquet(args.parquet, args.instance_id)
    else:
        if not args.issue_title or not args.issue_body:
            ap.error("--issue-title and --issue-body are required when using --repo")
        repo, issue_title, issue_body = args.repo, args.issue_title, args.issue_body

    with open(args.skillbook) as f:
        book = json.load(f)
    skills = book["skills"]
    print(f"Loaded {len(skills)} skills", file=sys.stderr)

    top_k = retrieve_top_k(
        skills,
        issue_title,
        issue_body,
        args.k,
        repo=repo,
        skillbook_path=args.skillbook,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        include_identifiers=args.include_identifiers,
    )

    print(f"\n{'=' * 80}")
    print(f"Repo:       {repo}")
    print(f"Title:      {issue_title}")
    print(f"Body:       {issue_body[:500]}{'...' if len(issue_body) > 500 else ''}")
    print(f"{'=' * 80}")

    for i, (skill_id, score) in enumerate(top_k, 1):
        skill = skills[skill_id]
        print(f"\n{i}. {skill_id}  (score: {score:.4f})")
        print(f"   Section:  {skill['section']}")
        print(f"   Content:  {skill['content']}")


if __name__ == "__main__":
    main()
