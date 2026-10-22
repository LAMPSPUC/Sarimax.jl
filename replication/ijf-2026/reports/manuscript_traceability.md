# Manuscript traceability

## The canonical revision is not available

The canonical manuscript is the revision of 2026-10-19 (controlled M4 objective-comparison
revision). Its three inputs are missing:

* `SARIMAX_jl_IJF_revision_20261019_controlled.pdf`
* `SARIMAX_jl_IJF_revision_20261019_controlled_supplement.pdf`
* `SARIMAX_jl_IJF_revision_20261019_controlled_source.zip`

They were searched for on 2026-10-20 in this repository and its branches, the artifact bundle,
the workspace that built it, the local downloads and documents folders and the project mailbox;
none was found. The transcription in `results/manuscript/manuscript_values.csv` is therefore
still that of the 2026-10-16 revision (672 values), and it was NOT relabelled as the 2026-10-19
one. `results/manuscript/REVISION.toml` records both revisions; the release check reports the
difference as missing evidence on every run.

## What was prepared for the 2026-10-19 revision

* **Mapping of the controlled items.** The 2026-10-19 revision takes Tables 5, 6, S9, S10,
  Figure 3 and the sample sizes of Section 5.4 from the controlled campaign (`m4_objectives`),
  not from the historical campaigns (`m4_objectives_icarai_*`).
  `results/manuscript/pending/manuscript_values_2026-10-19_controlled.csv` (written by
  `scripts/maintainers/make_controlled_mapping_template.py`) has one row per value those items
  printed in the 2026-10-16 layout (268), each pointing at the controlled cell holding the same
  statistic, with the printed value EMPTY. The values are entered from the 2026-10-19 source,
  never copied from the regenerated cells; until then every row is reported as "no evidence
  (awaiting transcription)", and each row with a value is compared like any other.
* **Regenerated sources for those items.** The table group `m4_objectives_controlled` rebuilds,
  from the per-series records of bundle part 2: `controlled.table5` (Table 5 and the Section 5.4
  sample sizes), `controlled.table6` (Table 6 and the panel counts of Figure 3),
  `controlled.paired` (new: the paired MASE and sMAPE quantiles of Tables S9 and S10),
  `controlled.quantiles` and `controlled.weekly`: 760 cells, all agreeing with the campaign's
  independent scorer.
* **The release check** (`scripts/maintainers/check_manuscript_mapping.py`, CI job "Manuscript
  release check") classifies every value: raw-reproduced, derived-only, differing, no evidence,
  plus the textual claims to verify (`results/manuscript/textual_claims.csv`). A differing value
  fails the check unless it has been established as an error of the manuscript and entered in
  `results/manuscript/errata.csv` with the archived value and the proposed correction; the
  release gate (`--release`, run before the image can be published) fails on registered errata
  too.

## Inventory of the 2026-10-16 transcription

| category | values | items |
|---|---|---|
| raw-reproduced | 603 | every regenerated row of Tables 2-7, S4, S8-S10, Figures 2-3 and Sections 5.3-5.6 except those below |
| derived-only | 56 | Tables S6 (9) and S7 (10), whose script and outputs are recovered and checked in CI; Table S5 (12), Section 5.6 (8), Table S2 (9) and Section 5.5 (8), checked against the records reconstructed on 2026-10-21 (`certification_batch_2026_08.md`) |
| differing | 6 | Table 7, Ipopt times 0.04 s and 0.02 s (`table7_timing_reconciliation.md`); Section 5.6, the four seasonal-grid figures (`certification_batch_2026_08.md`); all registered errata |
| no evidence | 5 | Section 5.5 (3: mixed seasonal orders 12.1% -> 16.5%, OWA +0.0097), Table S3 (2) (`provenance_gaps.md`) |
| checked at run time only | 2 | cells without a historical table (Section 5.3 range) |
| textual claims | 17 listed, 13 to verify | `results/manuscript/textual_claims.csv` |

"Raw-reproduced" in the static check means that the printed value agrees with the reference cell
that a table group regenerates from archived raw records; the regeneration itself is verified by
`./reproduce.sh tables` (every group agrees on every cell its inputs allow). Table 7's
certificate cells and objective column are counted as reproduced because they match the archived
records; what those records establish is qualified in `global_certification_audit.md`.

## Rebuilding the inventory for 2026-10-19 (when the source is supplied)

1. Unpack the source zip; transcribe every number of the tables, figures and experimental
   sections into a new `manuscript_values.csv`, keeping the 2026-10-16 file in the history.
   For the controlled items, start from the pending template and fill `printed`.
2. Re-check every other row against the new text (unchanged M4 baselines, Tables 2-4, 7, S2-S8,
   Sections 5.1-5.6), add rows for new numbers and remove rows for numbers no longer printed.
3. Set `[transcribed] revision = "2026-10-19"` in `REVISION.toml`, update `errata.csv` (an entry
   whose value the authors corrected becomes stale and fails the check until removed), and run
   `check_manuscript_mapping.py` and `./reproduce.sh tables` with the bundle.
4. Update the textual claims against the new wording.
