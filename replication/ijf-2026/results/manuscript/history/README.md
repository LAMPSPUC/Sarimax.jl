# Development records

Records of the replication package written before the manuscript was submitted. They are not
checked, are not part of the replication evidence, and no document of the package relies on
them; the submitted manuscript is described by `../README.md` and
`../../../verification/VERIFICATION_REPORT.md`.

| file | content |
|---|---|
| `draft-2026-10-16/manuscript_values.csv`, `manuscript_values_controlled_template.csv` | the transcription of a pre-submission draft and the source specification prepared for the objective-comparison items |
| `draft-2026-10-16/errata.csv`, `textual_claims.csv`, `REVISION.toml` | the findings registered against that draft and its identity |
| `draft-2026-10-16/FINAL_SCIENTIFIC_AUDIT.md`, `manuscript_traceability.md` | the audit notes written against that draft |

`scripts/maintainers/transcribe_manuscript.py` reads the two transcription files only to reuse
the source specification (source, row and column selectors) of values that the submitted
manuscript prints at the same position; every printed value is read from the submitted source.
