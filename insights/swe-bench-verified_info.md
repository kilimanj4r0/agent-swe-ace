# SWE-bench Verified

## Docker Images Summary

- **Dataset**: `princeton-nlp/SWE-bench_Verified`, split `test` — 500 instances
- **705 build logs** generated (env base images + per-instance images)
- **680/705 images built successfully** (96.5%)
- **25 failed builds** across 5 failure categories

## Build Environment

- **Base image**: `sweb.env.py.x86_64.c795f4b88616b8462021ed:latest` (Linux x86_64)
- **pip version in testbed**: 26.0.1 (Python 3.9)
- **swebench version**: 4.1.0
- **Key issue**: pip 26.0.1 removed `--no-use-pep517` flag and has stricter build isolation, breaking swebench's hardcoded install specs

## Failed Instances (25)

### 1. scikit-learn — `--no-use-pep517` removed from pip (10 instances)

`swebench/harness/constants/python.py` hardcodes `--no-use-pep517 --no-build-isolation` for scikit-learn versions 0.20–0.22 and 1.3–1.6. The flag was removed in pip 25.1.

| Instance | Version |
|---|---|
| `scikit-learn__scikit-learn-25102` | 1.4 |
| `scikit-learn__scikit-learn-25232` | 1.4 |
| `scikit-learn__scikit-learn-25500` | 1.3 |
| `scikit-learn__scikit-learn-25570` | 1.3 |
| `scikit-learn__scikit-learn-25638` | 1.3 |
| `scikit-learn__scikit-learn-25747` | 1.3 |
| `scikit-learn__scikit-learn-25931` | 1.5 |
| `scikit-learn__scikit-learn-25973` | 1.5 |
| `scikit-learn__scikit-learn-26194` | 1.5 |
| `scikit-learn__scikit-learn-26323` | 1.6 |

**Fix**: Replace `--no-use-pep517 --no-build-isolation` with `--no-build-isolation` in `SPECS_SKLEARN` (lines 23 and 39 of `swebench/harness/constants/python.py`).

### 2. astropy — `pkg_resources` missing in build isolation (2 instances)

`pip install -e .[test] --verbose` fails because `ah_bootstrap.py` does `import pkg_resources`. Pip 26's build isolation env uses a setuptools that doesn't expose `pkg_resources` at build time.

| Instance | Version |
|---|---|
| `astropy__astropy-8707` | 3.1 |
| `astropy__astropy-8872` | 3.1 |

**Fix**: Add `--no-build-isolation` to the install command in `SPECS_ASTROPY` for version 3.1, so it uses the already-installed `setuptools==68.0.0` (which has `pkg_resources`).

### 3. pylint-dev — missing `build_editable` hook (4 instances)

Old pylint build backend lacks PEP 660 `build_editable` support. `pip install -e .` fails because pip 26 defaults to editable installs via the build backend.

| Instance | Version |
|---|---|
| `pylint-dev__pylint-7114` | 2.15 |
| `pylint-dev__pylint-7228` | 2.15 |
| `pylint-dev__pylint-7277` | 2.15 |
| `pylint-dev__pylint-7993` | 2.15 |

**Fix**: Add `--no-build-isolation` to the install command in `SPECS_PYLINT` for version 2.15.

### 4. sympy-20590 — upstream branch removed (1 instance)

`swebench/harness/constants/__init__.py` maps commit `cffd4e0f...` to branch `1.7`, but that branch no longer exists in `github.com/sympy/sympy`. Only the tag `sympy-1.7` remains.

**Fix**: Change branch mapping from `"1.7"` to `"sympy-1.7"` in `REPO_BASE_COMMIT_BRANCH` (line 174 of `swebench/harness/constants/__init__.py`).

### 5. Transient network errors (8 instances) — retry should fix

Git clone failures during Docker build (`fetch-pack: invalid index-pack output`, `GnuTLS recv error`).

| Instance | Error |
|---|---|
| `django__django-13297` | fetch-pack: invalid index-pack output |
| `django__django-13363` | fetch-pack: invalid index-pack output |
| `django__django-13512` | fetch-pack: invalid index-pack output |
| `django__django-13809` | fetch-pack: invalid index-pack output |
| `django__django-14007` | fetch-pack: invalid index-pack output |
| `django__django-15315` | fetch-pack: invalid index-pack output |
| `django__django-16256` | fetch-pack: invalid index-pack output |
| `sympy__sympy-14711` | GnuTLS recv error (-110) |

**Fix**: Re-run `prepare_images.py --instances <ids>` for these 8. No code changes needed.

## Root Cause Summary

All deterministic failures stem from the base Docker image being rebuilt with **pip 26.0.1** while swebench 4.1.0 install specs target older pip behavior:

- `--no-use-pep517` flag removed
- Build isolation stricter (doesn't expose `pkg_resources`)
- Editable installs require PEP 660 `build_editable` hook by default

The fix pattern is consistent: add `--no-build-isolation` to install commands so they use the conda env's pre-installed packages instead of pip's isolated build environment.

## config.yaml exclude_instances (current)

8 instances already excluded from experiments. After fixing the root causes, only the 8 transient network failures need retry — no exclusions needed for those. New instances to consider excluding if not fixed:

- `pylint-dev__pylint-7277` (not in current exclude list)
- `scikit-learn__scikit-learn-25102` through `scikit-learn__scikit-learn-26323` (6 new scikit-learn instances)
- `astropy__astropy-8707`, `astropy__astropy-8872` (new)
