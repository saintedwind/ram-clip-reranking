# Publication audit â€” 2026-11-01

## Scope

The audited source and completed results already matched GitHub. Publication adds a review receipt, not a new model-performance experiment.

## Validation

33 Python regression tests passed again on 2026-11-01.

The published actual 512-image experiment and 8-image ontology smoke remain original dated results. Ontology annotation coverage increased from 88.02% to 96.09%; that is not model accuracy.

[Original evidence and limitations](VALIDATION_512.md)

## Limitations retained

Fusion/MMR did not beat the frozen RAM baseline in the completed evaluation. RAM-only intermediate traces and old notebook material are not promoted as new completed experiments.

## Publication boundary

Source and selected attributed evidence only. No credentials, private documents, raw downloaded evaluation inputs, model weights, local databases, caches or temporary runtime state are published. Historical experiments are not renamed as fresh validation. No production/customer accuracy claim is added.

Pre-publication main revision: `cbd8d529798dc377e043754cebd5511e16e10769`. GitHub Actions on the published source is the independent CI record; a historical green run is not substituted for a new code revision.
