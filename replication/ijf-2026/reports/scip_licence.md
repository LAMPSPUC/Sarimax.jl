# SCIP 8.0.0 in the replication image: licence and a possible migration

Not a legal opinion; a record of the facts for the maintainers' decision. Nothing was changed.

## Current state

* The replication environment pins SCIP.jl 0.11.6 with SCIP_jll 0.2.1+0, which ships SCIP
  8.0.0 (`environments/julia/Manifest.toml`; the image's build check prints "SCIP 8.0.0"). This
  is the SCIP build behind every archived SCIP record (the July 2026 battery) and behind the
  2026-10-20 re-run, so the environment reproduces the archived solver exactly.
* SCIP releases up to and including 8.0.2 are distributed under the ZIB Academic License; from
  8.0.3 (December 2022) SCIP is licensed under Apache 2.0 (scipopt.org, announcement of
  2022-11-04, read 2026-10-20).
* The ZIB Academic License is written for non-commercial academic use. A public container image
  on GHCR that contains the SCIP 8.0.0 binaries redistributes them to anyone; whether the
  licence allows that is the question to settle (with ZIB, or by the maintainers' reading of the
  licence) before `ijf-image-publish.yml` is run. The workflow's header already says so.

## Options

1. **Keep SCIP 8.0.0** and publish the image only after obtaining permission, or publish the
   image without SCIP and document that the SCIP experiments need a local build. The second
   option breaks `smoke` (it has a SCIP global solve) and the battery's SCIP runners in the
   published image.
2. **Migrate to an Apache-licensed SCIP within the same major version**: SCIP.jl 0.11.7 or later
   0.11.x accept SCIP_jll "800"; SCIP_jll 800.0.300 and later carry SCIP 8.0.3 or later (General
   registry, `S/SCIP/Compat.toml` and `jll/S/SCIP_jll/Versions.toml`, read 2026-10-20). This is
   the smallest change that removes the licence question.
3. **Migrate to SCIP 9 or 10** (SCIP.jl 0.12, SCIP_jll 900 or 1000): larger changes in the
   solver, no advantage for reproducibility.

## What a migration (option 2) would require

* Re-resolve the Julia environment with SCIP.jl 0.11.x (>= 0.11.7) and SCIP_jll >= 800.0.300,
  keeping every other package pinned; check that JuMP 1.31.1 and MathOptInterface 1.52.0 still
  resolve.
* Re-run, as NEW identified runs, everything that uses SCIP: the smoke test's SCIP global solve,
  `battery_solver_comparison` (MA(1) certificates), `battery_global_value` (Table 7),
  `battery_solver_scaling`; compare with the archived records and with the SCIP 8.0.0 re-run of
  2026-10-20. Solve times and, near the time limit, the certificates themselves can change
  (`global_certification_audit.md` §5.2); objectives and coefficients should agree to the
  solver's tolerance.
* Keep SCIP 8.0.0 documented as the archived solver: the manuscript's SCIP results were
  produced with it, and a migrated image would no longer re-run them with the same binary.

Not done here: the task fixes the solver at SCIP 8.0.0, and the migration needs a decision and
the re-runs above.
