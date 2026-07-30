#!/usr/bin/env python
"""Compact skills clustering (MiniLM or Qwen3-Embedding-4B).

Modes:
  --skillbook PATH                 single book (group label = section)
  --pool-dir DIR                   per_repo/*/final_skillbook.json (group = repo)
  --book PATH:TAG [PATH:TAG ...]   multiple books; per-book structure (group=section)
                                   + pooled cross-book comparison (group=tag).
                                   PATH may be a file (global book) or a directory
                                   (pools its per_repo/<repo>/final_skillbook.json).
                                   Model is loaded ONCE and embeddings reused per book.

For each unit it reports: natural k* (KMeans silhouette sweep), cluster-vs-label
purity/ARI, redundancy (cosine>=thr incl. cross-group), HDBSCAN, cluster representatives.
With --book it also prints a cross-tag near-duplicate matrix (semantic overlap across
the supplied books/backbones).

Usage:
  uv run python scripts/cluster_skills.py --skillbook <final_skillbook.json>
  uv run python scripts/cluster_skills.py --pool-dir <.../skillbooks/per_repo>
  uv run python scripts/cluster_skills.py --model Qwen/Qwen3-Embedding-4B \
      --book <q30_global.json>:Q30-glb-def <q30_perrepo_dir>:Q30-pr-def <qnext_global.json>:QNext-glb-def
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def load_file(path, tag):
    with open(path) as f:
        sb = json.load(f)
    sk = sb.get("skills", sb)
    if isinstance(sk, dict):
        sk = list(sk.values())
    return [{"id": s.get("id", ""), "section": s.get("section", ""),
             "content": s.get("content", ""), "tag": tag} for s in sk]


def load_path(path, tag):
    if os.path.isdir(path):
        out = []
        for p in sorted(glob.glob(os.path.join(path, "*/final_skillbook.json"))):
            out.extend(load_file(p, tag))
        return out
    return load_file(path, tag)


def gather(args):
    books = []
    if args.book:
        for spec in args.book:
            p, _, t = spec.partition(":")
            books.append((t or p, load_path(p, t or p)))
    elif args.pool_dir:
        for p in sorted(glob.glob(os.path.join(args.pool_dir, "*/final_skillbook.json"))):
            r = os.path.basename(os.path.dirname(p))
            books.append((r, load_file(p, r)))
    elif args.skillbook:
        books.append(("single", load_path(args.skillbook, "single")))
    skills = [s for _, sks in books for s in sks]
    return skills, [t for t, _ in books]


def analyze(name, emb, contents, group, gname, thr, mink, maxk, cross=False):
    from sklearn.cluster import KMeans, HDBSCAN
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    n = len(emb)
    sim = emb @ emb.T
    codes, ulabels = pdencode(group)
    iu = np.triu_indices(n, 1)
    pair = sim[iu]
    same = codes[iu[0]] == codes[iu[1]]
    ndup = int((pair >= thr).sum())
    ndup_x = int(((pair >= thr) & (~same)).sum())
    print(f"\n{'=' * 72}\n## {name}  (N={n}, {len(ulabels)} {gname}s)")
    print(f"  redundancy cos>={thr}: {ndup} near-dup pairs ({100*ndup/len(pair):.3f}%), "
          f"cross-{gname} {ndup_x}")
    print(f"  mean cos {pair.mean():.3f}  median {np.median(pair):.3f}  p95 {np.percentile(pair,95):.3f}")

    if cross and len(ulabels) > 1:
        m = len(ulabels)
        M = np.zeros((m, m), int)
        cmask = (pair >= thr) & (~same)
        ga, gb = codes[iu[0]][cmask], codes[iu[1]][cmask]
        np.add.at(M, (ga, gb), 1)
        np.add.at(M, (gb, ga), 1)
        print(f"  cross-{gname} near-dup matrix (cos>={thr}, upper=shared pairs):")
        hdr = "  " + " ".join(f"{t[:9]:>9}" for t in ulabels)
        print(hdr)
        for i, t in enumerate(ulabels):
            row = " ".join(f"{M[i,j]:>9}" for j in range(m))
            print(f"  {t[:9]:<9}{row}")

    # KMeans sweep (subsample silhouette + fewer inits for large N)
    sil_n = min(n, 2500)
    sub = np.random.RandomState(0).choice(n, sil_n, replace=False) if sil_n < n else np.arange(n)
    dist_sub = np.clip(1.0 - sim[np.ix_(sub, sub)], 0.0, None)  # cosine distance, >=0
    np.fill_diagonal(dist_sub, 0.0)
    n_init = 3 if n > 1500 else 10
    kmax = min(maxk, 25) if n > 1500 else maxk
    best = None
    for k in range(mink, min(kmax, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=0).fit(emb)
        if len(set(km.labels_)) < 2:
            continue
        sil = silhouette_score(dist_sub, km.labels_[sub], metric="precomputed")
        ari = adjusted_rand_score(group, km.labels_)
        if best is None or sil > best[1]:
            best = (k, sil, ari, km.labels_)
    kstar, sil, ari, lab = best
    pur = np.mean([Counter(group[i] for i in np.where(lab == c)[0]).most_common(1)[0][1]
                   / max(1, int((lab == c).sum())) for c in set(lab)])
    print(f"  KMeans k*={kstar} sil={sil:.3f} ARI(vs {gname})={ari:.3f} purity={pur:.3f}"
          + ("  [sil subsampled]" if sil_n < n else ""))
    h = HDBSCAN(min_cluster_size=5, metric="cosine").fit(emb)
    nclu = len(set(h.labels_)) - (1 if -1 in h.labels_ else 0)
    print(f"  HDBSCAN clusters={nclu} noise={100*(h.labels_==-1).mean():.1f}% "
          f"ARI(vs {gname})={adjusted_rand_score(group, h.labels_):.3f}")
    print(f"  top clusters (k*={kstar}):")
    for c, _ in Counter(lab.tolist()).most_common(10):
        idx = np.where(lab == c)[0]
        rep = idx[sim[np.ix_(idx, idx)].mean(1).argmax()]
        tc = Counter(group[i] for i in idx).most_common(1)[0]
        print(f"    cl{c}: n={len(idx)} top_{gname}={tc[0]}({tc[1]}/{len(idx)}) | \"{contents[rep][:82]}\"")


def pdencode(group):
    ulabels = sorted(set(group))
    idx = {t: i for i, t in enumerate(ulabels)}
    return np.array([idx[t] for t in group]), ulabels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skillbook")
    ap.add_argument("--pool-dir")
    ap.add_argument("--book", nargs="*", help="PATH:TAG (repeatable); PATH = file or per_repo dir")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min-k", type=int, default=4)
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--redundancy-thresh", type=float, default=0.85)
    args = ap.parse_args()
    if not (args.skillbook or args.pool_dir or args.book):
        ap.error("provide --skillbook, --pool-dir, or --book")

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mk = {"torch_dtype": torch.bfloat16} if device.startswith("cuda") else {}
    print(f"# model={args.model}  device={device}  thresh={args.redundancy_thresh}")
    model = SentenceTransformer(args.model, device=device, model_kwargs=mk)

    skills, book_tags = gather(args)
    n = len(skills)
    contents = [s["content"] for s in skills]
    sections = [s["section"] for s in skills]
    tags = [s["tag"] for s in skills]
    print(f"# pooled N={n}  books={book_tags}")

    emb = model.encode(contents, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)

    if args.book and len(book_tags) > 1:
        # per-book structure (group=section), reusing the single embedding pass
        for t in book_tags:
            idx = [i for i, s in enumerate(skills) if s["tag"] == t]
            analyze(t, emb[idx], [contents[i] for i in idx], [sections[i] for i in idx],
                    "section", args.redundancy_thresh, args.min_k, args.max_k)
        # pooled cross-book comparison (group=tag)
        analyze("POOLED cross-book", emb, contents, tags, "tag",
                args.redundancy_thresh, args.min_k, args.max_k, cross=True)
    else:
        group = tags if args.pool_dir else sections
        gname = "tag" if args.pool_dir else "section"
        analyze(args.skillbook or args.pool_dir or "book", emb, contents, group, gname,
                args.redundancy_thresh, args.min_k, args.max_k)


if __name__ == "__main__":
    main()
