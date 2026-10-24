# Printed values of the manuscript

`manuscript_values.csv` lists every number printed in the tables, figures and text of the
submitted manuscript and its Supplementary Material (836 values), with the label of its table,
figure or section and the archived source it is checked against. The mapping is bound to the
submitted files, not to a name: `REVISION.toml` holds the SHA-256 of the manuscript PDF, the
supplement PDF and the LaTeX sources.

**How it was built.** `scripts/maintainers/transcribe_manuscript.py --source DIR` reads the
printed values from the LaTeX source itself (table cells, figure annotations through
`pdftotext`, and numbers quoted in the text located by their exact LaTeX text) and pairs each
with its source; nothing was typed by hand. Run without `--write` on the same sources, it
reproduces this file byte for byte. Any change to the manuscript needs a new transcription and
new hashes in `REVISION.toml`.

**How it is checked.** `./reproduce.sh tables` compares each value with the value the
replication package regenerates and writes the cell-level result to
`output/reports/manuscript_cells.csv`. `scripts/maintainers/check_manuscript_mapping.py`
compares it, without the artifact bundle, with the historical value of the same cell or with
its derived record (CI job "Manuscript release check").

| file | content |
|---|---|
| `manuscript_values.csv` | the printed values and their sources |
| `derived_quantities.csv` | values that are simple functions of archived records but no archived table holds (Table 7 Ipopt times at the record's precision, Table S15's Stable-minus-MSE column, the Section 5.4 improvement shares at the declared tolerance), with the SHA-256 of each input (`scripts/maintainers/derive_manuscript_quantities.py`) |
| `textual_claims.csv` | statements of the text checked against the archive and the code, with their status (supported, differs, needs verification) |
| `errata.csv` | printed values established as errors of the manuscript; empty: no printed value differs from its source |
| `REVISION.toml` | the identity of the submitted manuscript |
| `history/` | transcriptions of pre-submission drafts, kept for development traceability; not checked and not part of the evidence |

| Column | Meaning |
|---|---|
| `item`, `label` | where the value is printed (e.g. Table 3, `tab:m4_forecasting`) |
| `row`, `column` | its position, in the manuscript's own words |
| `printed`, `decimals` | the value as printed and its precision (decimal places; for a value printed as `a.bc × 10^-k`, the decimal places of the equivalent fixed-point number) |
| `scale` | factor applied to the source before comparing (100 for percentages) |
| `source_kind` | `regenerated` (a cell of a table group, rebuilt from per-series or per-instance records) or `derived` (a row of an archived record in the repository that no table group rebuilds: a value recomputed by a check from an identified re-run, a recovered output, a derived quantity, or a summary record) |
| `source`, `source_row`, `source_column` | the cell compared with: `<group>/<table>` and the cell keys, or a CSV path and `column=value` selectors |
| `note` | provenance remarks |

A value agrees when the source, rounded to the printed precision, gives the printed value
(|source - printed| <= half a unit of the last printed decimal); a text cell is compared on its
first word. A `derived` value is never counted as reproduced.
