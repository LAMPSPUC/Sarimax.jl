# Printed values of the manuscript

`manuscript_values.csv` lists every number printed in the tables, figures and experimental
sections of the manuscript, with the LaTeX label of its table, figure or section and the
source it is checked against. `./reproduce.sh tables` compares each one with the value the
replication package regenerates and writes the cell-level result to
`output/reports/manuscript_cells.csv`; `scripts/maintainers/check_manuscript_mapping.py`
compares it, without the artifact bundle, with the historical reference tables.

**Version.** The values are those of the revision of 2026-10-16 (main text and Supplementary
Material). They were parsed from the LaTeX source of the 2026-10-11 revision, and every
transcribed table row was checked to appear in the text of the 2026-10-16 PDF; Table S4 and
the panel percentages of Figure 2, which are not in that source, were read from the
2026-10-16 PDF. When the manuscript changes, this file must be updated with it.

| Column | Meaning |
|---|---|
| `item`, `label` | where the value is printed (e.g. Table 3, `tab:m4_forecasting`) |
| `row`, `column` | its position, in the manuscript's own words |
| `printed`, `decimals` | the value as printed and its number of decimals (empty for text) |
| `scale` | factor applied to the source before comparing (100 for percentages) |
| `source_kind` | `regenerated` (a cell of a table group), `derived` (a derived table archived in `results/historical/`, whose raw inputs are not archived), or `none` (no archived source identified; `note` says why) |
| `source`, `source_row`, `source_column` | the cell compared with: `<group>/<table>` and the cell keys, or a CSV path and `column=value` selectors |
| `note` | provenance remarks, such as protocol differences between the text and the archived run |

A value agrees when the source, rounded to the printed precision, gives the printed value
(|source - printed| <= half a unit of the last printed decimal); a text cell is compared on its
first word. The comparison never treats `derived` or `none` values as reproduced.
