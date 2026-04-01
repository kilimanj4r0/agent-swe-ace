# SWE-bench Lite

## Docker Images Summary

- **329 sweb images** (eval + env bases + 1 stray `swebench/` image)
- **12 projects**: django(114), sympy(76), matplotlib(23), scikit-learn(19), pytest-dev(17), sphinx-doc(16), astropy(12), psf(6), pydata(5), mwaskom(4), pylint-dev(3), pallets(3)
- **Virtual size**: ~1,058 GB
- **Actual on-disk**: ~2.7 GB (99.7% layers shared across images)
- Non-sweb images (Opik, Memgraph, clickhouse, mysql, redis, minio, ubuntu): ~138 GB
- **Total Docker images**: 347, 140.8 GB on-disk, 134.6 GB reclaimable

## Excluded Instances

7 instances fail to build due to upstream issues (not fixable by retry).
292/300 images built successfully (97.3%).

## Excluded Instances

| Instance | Error | Cause | Reference |
|---|---|---|---|
| `pylint-dev__pylint-7114` | pip install -e . fails | astroid version conflict | [SWE-agent#690](https://github.com/SWE-agent/SWE-bench/issues/690) |
| `pylint-dev__pylint-7228` | pip install -e . fails | astroid version conflict | SWE-agent#690 |
| `pylint-dev__pylint-7993` | pip install -e . fails | astroid version conflict | SWE-agent#690 |
| `scikit-learn__scikit-learn-25500` | no such option: --no-use-pep517 | Flag removed in pip 24.1 | Downgrade pip or use --no-build-isolation |
| `scikit-learn__scikit-learn-25570` | no such option: --no-use-pep517 | Flag removed in pip 24.1 | |
| `scikit-learn__scikit-learn-25747` | no such option: --no-use-pep517 | Flag removed in pip 24.1 | |
| `scikit-learn__scikit-learn-25638` | no such option: --no-use-pep517 | Flag removed in pip 24.1 | |
| `sympy__sympy-20590` | Remote branch not found | Branch deleted on GitHub | [SWE-bench#167](https://github.com/princeton-nlp/SWE-bench/issues/167) |

## Notes

- pylint/scikit-learn/sympy failures are dataset-level issues tracked upstream.
